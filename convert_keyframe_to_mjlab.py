#!/usr/bin/env python3
"""Convert keyframe joint positions from keyframe app to mjlab HOME_KEYFRAME format.

The keyframe app shows ALL joints including driven joints (mechanism joints).
mjlab HOME_KEYFRAME should only set ACTUATED joints (drive joints).

Equality constraints in the XML define: driven = coef × drive
So we need: drive = driven / coef

From toddlerbot_2xc_original.xml equality section:
- neck_yaw_driven = -0.9090909091 × neck_yaw_drive
- left/right_hip_yaw_driven = -0.8571428571 × hip_yaw_drive
- shoulder_yaw_driven = -1.0 × shoulder_yaw_drive
- elbow_yaw_driven = -1.0 × elbow_yaw_drive
- wrist_pitch_driven = -1.0 × wrist_pitch_drive
"""

import sys


# Mapping: (driven_joint_name, drive_joint_name, coefficient)
DRIVEN_TO_DRIVE_MAPPING = [
    ("neck_yaw_driven", "neck_yaw_drive", -0.9090909091),
    ("left_hip_yaw_driven", "left_hip_yaw_drive", -0.8571428571),
    ("right_hip_yaw_driven", "right_hip_yaw_drive", -0.8571428571),
    ("left_shoulder_yaw_driven", "left_shoulder_yaw_drive", -1.0),
    ("right_shoulder_yaw_driven", "right_shoulder_yaw_drive", -1.0),
    ("left_elbow_yaw_driven", "left_elbow_yaw_drive", -1.0),
    ("right_elbow_yaw_driven", "right_elbow_yaw_drive", -1.0),
    ("left_wrist_pitch_driven", "left_wrist_pitch_drive", -1.0),
    ("right_wrist_pitch_driven", "right_wrist_pitch_drive", -1.0),
]

# Joints that are NOT actuated (driven/passive mechanism joints)
NON_ACTUATED_JOINTS = {
    "neck_yaw_driven",
    "neck_pitch",
    "neck_pitch_front",
    "neck_pitch_back",
    "waist_yaw",
    "waist_roll",
    "left_hip_yaw_driven",
    "right_hip_yaw_driven",
    "left_shoulder_yaw_driven",
    "right_shoulder_yaw_driven",
    "left_elbow_yaw_driven",
    "right_elbow_yaw_driven",
    "left_wrist_pitch_driven",
    "right_wrist_pitch_driven",
}


def convert_keyframe_positions(joint_positions: dict[str, float]) -> dict[str, float]:
    """Convert joint positions from keyframe app to mjlab format.
    
    Args:
        joint_positions: Dict of ALL joint positions (including driven joints)
        
    Returns:
        Dict of ACTUATED joint positions for mjlab HOME_KEYFRAME
    """
    actuated_positions = {}
    
    # First, convert driven joints to drive joints
    for driven_name, drive_name, coef in DRIVEN_TO_DRIVE_MAPPING:
        if driven_name in joint_positions:
            # driven = coef × drive, so drive = driven / coef
            driven_value = joint_positions[driven_name]
            drive_value = driven_value / coef
            actuated_positions[drive_name] = drive_value
            print(f"  {driven_name:30s} = {driven_value:8.4f} → {drive_name:30s} = {drive_value:8.4f}")
    
    # Then, copy over all other actuated joints
    for joint_name, value in joint_positions.items():
        # Skip non-actuated joints
        if joint_name in NON_ACTUATED_JOINTS:
            continue
        # Skip if already converted from driven
        if joint_name not in actuated_positions:
            actuated_positions[joint_name] = value
    
    return actuated_positions


def print_home_keyframe_code(joint_positions: dict[str, float], base_pos: tuple[float, float, float] = (0.0, 0.0, 0.310053)):
    """Print the HOME_KEYFRAME code for mjlab constants file."""
    
    print("\n" + "="*70)
    print("Copy this into toddlerbot_2xc_constants.py:")
    print("="*70)
    print()
    print("HOME_KEYFRAME = EntityCfg.InitialStateCfg(")
    print(f"    pos={base_pos},")
    print("    joint_pos={")
    
    # Group joints by category
    neck_joints = {k: v for k, v in joint_positions.items() if 'neck' in k}
    waist_joints = {k: v for k, v in joint_positions.items() if 'waist' in k}
    left_leg = {k: v for k, v in joint_positions.items() if 'left' in k and any(x in k for x in ['hip', 'knee', 'ankle'])}
    right_leg = {k: v for k, v in joint_positions.items() if 'right' in k and any(x in k for x in ['hip', 'knee', 'ankle'])}
    left_arm = {k: v for k, v in joint_positions.items() if 'left' in k and any(x in k for x in ['shoulder', 'elbow', 'wrist'])}
    right_arm = {k: v for k, v in joint_positions.items() if 'right' in k and any(x in k for x in ['shoulder', 'elbow', 'wrist'])}
    
    if neck_joints:
        print("        # Neck joints")
        for k, v in sorted(neck_joints.items()):
            print(f'        "{k}": {v},')
    
    if waist_joints:
        print("\n        # Waist joints")
        for k, v in sorted(waist_joints.items()):
            print(f'        "{k}": {v},')
    
    if left_leg:
        print("\n        # Left leg joints")
        for k, v in sorted(left_leg.items()):
            print(f'        "{k}": {v},')
    
    if right_leg:
        print("\n        # Right leg joints")
        for k, v in sorted(right_leg.items()):
            print(f'        "{k}": {v},')
    
    if left_arm:
        print("\n        # Left arm joints")
        for k, v in sorted(left_arm.items()):
            print(f'        "{k}": {v},')
    
    if right_arm:
        print("\n        # Right arm joints")
        for k, v in sorted(right_arm.items()):
            print(f'        "{k}": {v},')
    
    print('\n        ".*": 0.0,  # Default for any other joints')
    print("    },")
    print('    joint_vel={".*": 0.0},')
    print("    ctrl={")
    print("        # Control targets match joint positions")
    for k, v in sorted(joint_positions.items()):
        print(f'        "{k}": {v},')
    print('        ".*": 0.0,')
    print("    },")
    print(")")
    print()
    print("="*70)


if __name__ == "__main__":
    print("ToddlerBot Keyframe Converter")
    print("="*70)
    print()
    print("This script converts joint positions from the keyframe app")
    print("(which shows driven joints) to mjlab format (drive joints only).")
    print()
    
    # Example: Using the viser slider values you provided
    example_keyframe = {
        # From your viser sliders (these are DRIVEN joint values)
        "left_hip_pitch": -0.4136,
        "left_hip_roll": -0.0031,
        "left_hip_yaw_driven": -0.0063,  # DRIVEN joint
        "left_knee": -0.9527,
        "left_ankle_roll": 0.0014,
        "left_ankle_pitch": -0.5421,
        
        "right_hip_pitch": 0.4158,
        "right_hip_roll": -0.0014,
        "right_hip_yaw_driven": 0.0031,  # DRIVEN joint
        "right_knee": 0.9562,
        "right_ankle_roll": -0.0065,
        "right_ankle_pitch": 0.5424,
        
        "left_shoulder_pitch": 0.1736,
        "left_shoulder_roll": 0.0713,
        "left_shoulder_yaw_driven": -1.5621,  # DRIVEN joint
        "left_elbow_roll": -0.5135,
        "left_elbow_yaw_driven": 1.5692,  # DRIVEN joint
        "left_wrist_pitch_driven": 1.2209,  # DRIVEN joint
        "left_wrist_roll": 0.0,
        
        "right_shoulder_pitch": -0.1736,
        "right_shoulder_roll": 0.0808,
        "right_shoulder_yaw_driven": 1.5568,  # DRIVEN joint
        "right_elbow_roll": -0.5133,
        "right_elbow_yaw_driven": -1.5691,  # DRIVEN joint
        "right_wrist_pitch_driven": -1.2209,  # DRIVEN joint
        "right_wrist_roll": 0.0,
        
        "neck_yaw_drive": 0.0,
        "neck_pitch_act": 0.0,
        "waist_act_1": 0.0,
        "waist_act_2": 0.0,
    }
    
    print("Converting driven joint values to drive joint values:")
    print("-" * 70)
    actuated = convert_keyframe_positions(example_keyframe)
    
    print()
    print_home_keyframe_code(actuated)

