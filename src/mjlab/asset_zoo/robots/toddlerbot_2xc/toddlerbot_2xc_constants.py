"""ToddlerBot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

TODDLERBOT_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "toddlerbot_2xc"
  / "xmls"
  / "toddlerbot_2xc.xml"
)
assert TODDLERBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, TODDLERBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TODDLERBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

# ORIGINAL POSE (slight bend, upright stance):
# To revert, uncomment below and comment out the current HOME_KEYFRAME
"""
HOME_KEYFRAME_ORIGINAL = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.310053),
    joint_pos={
        r"^left_hip_pitch$": -0.091312,
        r"^left_knee$": -0.380812,
        r"^left_ankle_pitch$": -0.2895,
        r"^right_hip_pitch$": 0.091312,
        r"^right_knee$": 0.380812,
        r"^right_ankle_pitch$": 0.2895,
        r"^left_shoulder_pitch$": 0.174533,
        r"^left_shoulder_roll$": 0.087266,
        r"^left_shoulder_yaw_drive$": 1.570796,
        r"^left_elbow_roll$": -0.523599,
        r"^left_elbow_yaw_drive$": -1.570796,
        r"^left_wrist_pitch_drive$": -1.22173,
        r"^right_shoulder_pitch$": -0.174533,
        r"^right_shoulder_roll$": 0.087266,
        r"^right_shoulder_yaw_drive$": -1.570796,
        r"^right_elbow_roll$": -0.523599,
        r"^right_elbow_yaw_drive$": 1.570796,
        r"^right_wrist_pitch_drive$": 1.22173,
        # Driven joints (computed from drive via equality constraints)
        r"^left_shoulder_yaw_driven$": -1.570796,
        r"^left_elbow_yaw_driven$": 1.570796,
        r"^left_wrist_pitch_driven$": 1.22173,
        r"^right_shoulder_yaw_driven$": 1.570796,
        r"^right_elbow_yaw_driven$": -1.570796,
        r"^right_wrist_pitch_driven$": -1.22173,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
    ctrl={
        r"^left_hip_pitch$": -0.091312,
        r"^left_knee$": -0.380812,
        r"^left_ankle_pitch$": -0.2895,
        r"^right_hip_pitch$": 0.091312,
        r"^right_knee$": 0.380812,
        r"^right_ankle_pitch$": 0.2895,
        r"^left_shoulder_pitch$": 0.174533,
        r"^left_shoulder_roll$": 0.087266,
        r"^left_shoulder_yaw_drive$": 1.570796,
        r"^left_elbow_roll$": -0.523599,
        r"^left_elbow_yaw_drive$": -1.570796,
        r"^left_wrist_pitch_drive$": -1.22173,
        r"^right_shoulder_pitch$": -0.174533,
        r"^right_shoulder_roll$": 0.087266,
        r"^right_shoulder_yaw_drive$": -1.570796,
        r"^right_elbow_roll$": -0.523599,
        r"^right_elbow_yaw_drive$": 1.570796,
        r"^right_wrist_pitch_drive$": 1.22173,
    },
)
"""

# CURRENT POSE (knee-bending, lower CoM for stability):
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.310053),  # Base position
    joint_pos={
        # Use regex anchors (^$) for exact matching to avoid "drive" matching "driven"!
        # Set BOTH drive and driven joints to satisfy equality constraints in keyframe.
        # Stiff equality constraints (in XML) will maintain coupling during dynamics.
        
        # Driven joints (from keyframe app sliders)
        r"^left_hip_yaw_driven$": -0.0063,
        r"^left_shoulder_yaw_driven$": -1.5621,
        r"^left_elbow_yaw_driven$": 1.5692,
        r"^left_wrist_pitch_driven$": 1.2209,
        r"^right_hip_yaw_driven$": 0.0031,
        r"^right_shoulder_yaw_driven$": 1.568,
        r"^right_elbow_yaw_driven$": -1.5691,
        r"^right_wrist_pitch_driven$": -1.2209,
        r"^neck_yaw_driven$": 0.0,
        
        # Mechanism joints
        r"^neck_pitch$": 0.0,
        r"^neck_pitch_front$": 0.0,
        r"^neck_pitch_back$": 0.0,
        r"^waist_yaw$": 0.0,
        r"^waist_roll$": 0.0,
        
        # Actuated joints
        r"^neck_yaw_drive$": 0.0,
        r"^neck_pitch_act$": 0.0,
        r"^waist_act_1$": 0.0,
        r"^waist_act_2$": 0.0,
        
        # Left leg (knee-bending pose from keyframe app)
        r"^left_hip_pitch$": -0.4136,
        r"^left_hip_roll$": -0.0031,
        r"^left_hip_yaw_drive$": 0.00735,  # = -0.0063 / -0.857
        r"^left_knee$": -0.9527,
        r"^left_ankle_roll$": 0.0014,
        r"^left_ankle_pitch$": -0.5421,
        
        # Right leg (knee-bending pose from keyframe app)
        r"^right_hip_pitch$": 0.4158,
        r"^right_hip_roll$": -0.0014,
        r"^right_hip_yaw_drive$": -0.00362,  # = 0.0031 / -0.857
        r"^right_knee$": 0.9562,
        r"^right_ankle_roll$": -0.0065,
        r"^right_ankle_pitch$": 0.5424,
        
        # Left arm (drive joints computed from driven values)
        r"^left_shoulder_pitch$": 0.1736,
        r"^left_shoulder_roll$": 0.0713,
        r"^left_shoulder_yaw_drive$": 1.5621,  # = -1.5621 / -1.0
        r"^left_elbow_roll$": -0.5135,
        r"^left_elbow_yaw_drive$": -1.5692,  # = 1.5692 / -1.0
        r"^left_wrist_pitch_drive$": -1.2209,  # = 1.2209 / -1.0
        r"^left_wrist_roll$": 0.0,
        
        # Right arm (drive joints computed from driven values)
        r"^right_shoulder_pitch$": -0.1736,
        r"^right_shoulder_roll$": 0.0808,
        r"^right_shoulder_yaw_drive$": -1.568,  # = 1.568 / -1.0
        r"^right_elbow_roll$": -0.5133,
        r"^right_elbow_yaw_drive$": 1.5691,  # = -1.5691 / -1.0
        r"^right_wrist_pitch_drive$": 1.2209,  # = -1.2209 / -1.0
        r"^right_wrist_roll$": 0.0,
        
        # Default for any remaining unspecified joints
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},  # All joint velocities set to zero
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},  # TODO: tune friction
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={r".*_collision": 1},  # Enable collision with terrain for all geoms
  conaffinity=0,  # Disable self-collision (geoms won't collide with each other)
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Actuator config.
##

# Define actuators for controllable joints only (excluding mechanism joints like *_driven)
# CRITICAL: Use regex anchors (^$) to prevent "drive" from matching "driven"!
# Without anchors, "left_hip_yaw_drive" matches BOTH "left_hip_yaw_drive" AND "left_hip_yaw_driven"
TODDLERBOT_ACTUATORS = BuiltinPositionActuatorCfg(
  joint_names_expr=(
    # CRITICAL: Order must match EXACTLY the original XML actuator section!
    # Neck
    r"^neck_yaw_drive$",
    r"^neck_pitch_act$",
    # Waist
    r"^waist_act_1$",
    r"^waist_act_2$",
    # Left leg - NOTE: ankle_roll BEFORE ankle_pitch (matches original XML!)
    r"^left_hip_pitch$",
    r"^left_hip_roll$",
    r"^left_hip_yaw_drive$",
    r"^left_knee$",
    r"^left_ankle_roll$",
    r"^left_ankle_pitch$",
    # Right leg - NOTE: ankle_roll BEFORE ankle_pitch
    r"^right_hip_pitch$",
    r"^right_hip_roll$",
    r"^right_hip_yaw_drive$",
    r"^right_knee$",
    r"^right_ankle_roll$",
    r"^right_ankle_pitch$",
    # Left arm
    r"^left_shoulder_pitch$",
    r"^left_shoulder_roll$",
    r"^left_shoulder_yaw_drive$",
    r"^left_elbow_roll$",
    r"^left_elbow_yaw_drive$",
    r"^left_wrist_pitch_drive$",
    r"^left_wrist_roll$",
    # Right arm
    r"^right_shoulder_pitch$",
    r"^right_shoulder_roll$",
    r"^right_shoulder_yaw_drive$",
    r"^right_elbow_roll$",
    r"^right_elbow_yaw_drive$",
    r"^right_wrist_pitch_drive$",
    r"^right_wrist_roll$",
  ),
  effort_limit=100.0,  # N·m, tune this based on actual motor specs
  armature=0.01,  # kg·m², tune based on actual motor specs
  stiffness=100.0,  # N·m/rad, tune for desired impedance
  damping=10.0,  # N·m·s/rad, tune for desired damping
)

##
# Final config.
##

TODDLERBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(TODDLERBOT_ACTUATORS,),
  soft_joint_pos_limit_factor=0.9,
)

TODDLERBOT_ROBOT_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(FULL_COLLISION,),  # FULL_COLLISION causes segfault with MJWarp
  spec_fn=get_spec,
  articulation=TODDLERBOT_ARTICULATION,
)

TODDLERBOT_ACTION_SCALE = {".*": 0.25}


def get_toddlerbot_robot_cfg() -> EntityCfg:
  """Get the ToddlerBot robot configuration.

  Returns a fresh copy to allow per-instance customization.
  """
  from dataclasses import replace

  return replace(TODDLERBOT_ROBOT_CFG)

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(TODDLERBOT_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
