#!/usr/bin/env python3
"""Quick script to visualize ToddlerBot's default pose in viser."""

import time
import numpy as np
import viser
import viser.transforms as vtf

from src.mjlab.asset_zoo.robots.toddlerbot_2xc.toddlerbot_2xc_constants import (
    TODDLERBOT_ROBOT_CFG,
    HOME_KEYFRAME,
)
from src.mjlab.entity.entity import Entity
from src.mjlab.viewer.viser_conversions import mujoco_mesh_to_trimesh


def visualize_toddlerbot_default_pose():
    """Visualize ToddlerBot in its default/home pose using viser."""
    
    print("🤖 Loading ToddlerBot...")
    robot = Entity(TODDLERBOT_ROBOT_CFG)
    mj_model = robot.spec.compile()
    
    # Create mujoco data
    import mujoco
    mj_data = mujoco.MjData(mj_model)
    
    print(f"✓ ToddlerBot loaded successfully")
    print(f"  Bodies: {mj_model.nbody}")
    print(f"  Joints: {mj_model.njnt}")
    print(f"  DOFs: {mj_model.nv}")
    print(f"  Collision geoms: {sum(1 for i in range(mj_model.ngeom) if mj_model.geom_contype[i] != 0)}")
    
    # Initialize to home pose using HOME_KEYFRAME configuration
    mujoco.mj_resetData(mj_model, mj_data)
    
    # Set base position (first 3 elements of qpos for free joint)
    mj_data.qpos[0:3] = HOME_KEYFRAME.pos
    
    # Set base quaternion (next 4 elements) - identity quaternion (no rotation)
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    
    # Apply HOME_KEYFRAME joint positions
    print("\n📐 Setting joint positions:")
    for joint_name, joint_pos in HOME_KEYFRAME.joint_pos.items():
        # Skip regex patterns
        if '.*' in joint_name or '*' in joint_name:
            continue
            
        try:
            # Find joint id by name
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = mj_model.jnt_qposadr[joint_id]
                mj_data.qpos[qpos_addr] = joint_pos
                print(f"  ✓ {joint_name}: {joint_pos:.4f} rad")
            else:
                print(f"  ✗ Joint not found: {joint_name}")
        except Exception as e:
            print(f"  ✗ Error setting {joint_name}: {e}")
    
    # Ensure velocities are zero
    mj_data.qvel[:] = 0
    
    # Forward kinematics to compute body positions
    mujoco.mj_forward(mj_model, mj_data)
    
    print("\n🌐 Starting viser server...")
    server = viser.ViserServer(label="ToddlerBot Default Pose")
    print(f"✓ Viser server running at http://localhost:8080")
    
    # Set up lighting
    server.scene.configure_environment_map(environment_intensity=0.8)
    
    # Add ground plane for reference
    server.scene.add_grid(
        "/ground",
        width=2.0,
        height=2.0,
        width_segments=20,
        height_segments=20,
        plane="xz",
    )
    
    print("\n📊 Adding robot visualization...")
    
    # Visualize each body
    body_handles = []
    for body_id in range(1, mj_model.nbody):  # Skip world body (0)
        # Get geoms for this body
        body_geom_ids = [i for i in range(mj_model.ngeom) 
                        if mj_model.geom_bodyid[i] == body_id]
        
        if not body_geom_ids:
            continue
            
        # Get body name
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            body_name = f"body_{body_id}"
        
        # Merge geoms for this body
        meshes = []
        for geom_id in body_geom_ids:
            # Only visualize visual geoms (group 2)
            if mj_model.geom_group[geom_id] != 2:
                continue
            
            # Skip non-mesh geoms (primitives like box, sphere, etc.)
            geom_type = mj_model.geom_type[geom_id]
            if geom_type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            
            try:
                mesh = mujoco_mesh_to_trimesh(mj_model, int(geom_id))
                if mesh is not None:
                    meshes.append(mesh)
            except Exception as e:
                print(f"Warning: Could not convert geom {geom_id}: {e}")
                continue
        
        if not meshes:
            continue
            
        # Merge all meshes for this body
        import trimesh
        combined_mesh = trimesh.util.concatenate(meshes)
        
        # Get body pose
        body_pos = mj_data.xpos[body_id]
        body_mat = mj_data.xmat[body_id].reshape(3, 3)
        body_quat = vtf.SO3.from_matrix(body_mat).wxyz
        
        # Add to viser
        handle = server.scene.add_mesh_simple(
            f"/robot/{body_name}",
            vertices=combined_mesh.vertices,
            faces=combined_mesh.faces,
            position=body_pos,
            wxyz=body_quat,
            color=(0.8, 0.8, 0.8),
        )
        body_handles.append(handle)
    
    print(f"✓ Added {len(body_handles)} body meshes")
    
    # Add coordinate frame at base
    server.scene.add_frame(
        "/robot/base_frame",
        axes_length=0.1,
        axes_radius=0.005,
    )
    
    # Print joint positions
    print("\n📐 Default joint positions:")
    for joint_id in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name and mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
            qpos_idx = mj_model.jnt_qposadr[joint_id]
            print(f"  {joint_name:30s}: {mj_data.qpos[qpos_idx]:8.4f} rad ({np.degrees(mj_data.qpos[qpos_idx]):7.2f}°)")
    
    print("\n✅ Visualization ready!")
    print("   Open http://localhost:8080 in your browser")
    print("   Press Ctrl+C to exit\n")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.stop()


if __name__ == "__main__":
    visualize_toddlerbot_default_pose()

