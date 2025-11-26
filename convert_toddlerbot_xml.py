#!/usr/bin/env python3
"""
Script to convert ToddlerBot XML to be compatible with mjlab framework.

This script:
1. Removes XML-defined actuators (framework uses programmatic ActuatorCfg)
2. Comments out explicit contact pairs (they conflict with MJWarp collision kernels)
3. Preserves equality constraints (needed for gear mechanisms like neck pitch)

The main issue: MJWarp segfaults when explicit <contact><pair> elements are combined
with FULL_COLLISION collision configuration. Commenting them out allows the framework
to handle collisions automatically.

Usage:
    python convert_toddlerbot_xml.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from datetime import datetime


def convert_toddlerbot_xml():
    """Convert ToddlerBot XML to framework-compatible version."""
    
    # Paths
    original_xml = Path("src/mjlab/asset_zoo/robots/toddlerbot_2xc/xmls/toddlerbot_2xc.xml")
    backup_xml = Path("src/mjlab/asset_zoo/robots/toddlerbot_2xc/xmls/toddlerbot_2xc_original.xml")
    
    if not original_xml.exists():
        print(f"❌ Original XML not found: {original_xml}")
        return False
        
    print(f"🔄 Converting {original_xml} to framework-compatible version...")
    
    # Create backup
    if not backup_xml.exists():
        shutil.copy2(original_xml, backup_xml)
        print(f"✅ Created backup: {backup_xml}")
    
    # Parse XML
    tree = ET.parse(original_xml)
    root = tree.getroot()
    
    # Track changes
    changes = []
    
    # 1. Remove all actuators (framework will add them programmatically via ActuatorCfg)
    actuator_section = root.find('actuator')
    if actuator_section is not None:
        num_actuators = len(actuator_section.findall('motor'))
        root.remove(actuator_section)
        changes.append(f"Removed {num_actuators} XML-defined actuators (will use programmatic ActuatorCfg)")
    
    # 2. Keep equality constraints - they're essential for gear mechanisms (e.g., neck pitch linkage)
    equality_section = root.find('equality')
    if equality_section is not None:
        num_constraints = len(list(equality_section))
        changes.append(f"Preserved {num_constraints} equality constraints (needed for gear mechanisms)")
    
    # 3. Comment out explicit contact pairs - they conflict with MJWarp collision detection
    # When explicit <pair> elements exist, MJWarp segfaults during ccd_kernel compilation
    # with FULL_COLLISION. The framework handles collisions better automatically.
    contact_section = root.find('contact')
    if contact_section is not None:
        num_pairs = len(list(contact_section))
        
        # Create comment with the contact section
        contact_str = ET.tostring(contact_section, encoding='unicode')
        contact_comment = ET.Comment(
            f' Commenting out explicit contact pairs - they conflict with MJWarp when using FULL_COLLISION\n'
            f'{contact_str}\n'
            f'  '
        )
        
        # Find position of contact section and replace with comment
        parent = root
        for i, child in enumerate(parent):
            if child == contact_section:
                parent.remove(contact_section)
                parent.insert(i, contact_comment)
                break
        
        changes.append(f"Commented out {num_pairs} explicit contact pairs (conflict with MJWarp collision kernels)")
    
    # 4. Remove keyframe with control values (framework will handle this)
    keyframe_section = root.find('keyframe')
    if keyframe_section is not None:
        for key in keyframe_section.findall('key'):
            if key.get('ctrl') is not None:
                key.attrib.pop('ctrl', None)
                changes.append("Removed control values from keyframe")
    
    # 5. Add comment indicating this is a converted file
    comment = ET.Comment(f' Converted for mjlab framework compatibility on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ')
    root.insert(0, comment)
    
    # Write the modified XML
    tree.write(original_xml, encoding='utf-8', xml_declaration=True)
    
    # Pretty print the changes
    print("\n📋 Changes made:")
    for i, change in enumerate(changes, 1):
        print(f"  {i}. {change}")
    
    print(f"\n✅ Conversion complete! Modified XML saved to: {original_xml}")
    print(f"💾 Original XML backed up to: {backup_xml}")
    
    return True


def create_compatible_constants():
    """Update the constants file to work with the converted XML."""
    
    constants_file = Path("src/mjlab/asset_zoo/robots/toddlerbot_2xc/toddlerbot_2xc_constants.py")
    
    print(f"\n🔄 Updating constants file: {constants_file}")
    
    # Read current file
    with open(constants_file, 'r') as f:
        content = f.read()
    
    # Check if already has the new format
    if 'TODDLERBOT_ACTUATORS' in content and 'EntityArticulationInfoCfg' in content:
        print("✅ Constants file already has actuator configuration")
        return
    
    # Add necessary imports if not present
    if 'EntityArticulationInfoCfg' not in content:
        content = content.replace(
            'from mjlab.entity import EntityCfg',
            'from mjlab.entity import EntityArticulationInfoCfg, EntityCfg'
        )
    if 'ActuatorCfg' not in content:
        content = content.replace(
            'from mjlab.utils.spec_config import CollisionCfg',
            'from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg'
        )
    
    # Create new actuator configuration matching the original XML
    # Using a single ActuatorCfg for all controllable joints (excluding mechanism joints)
    new_actuator_config = '''##
# Actuator config.
##

# Define actuators for controllable joints only (excluding mechanism joints like *_driven)
# Based on the original XML actuator definitions
# These values are estimates - tune them based on actual Dynamixel motor specs
TODDLERBOT_ACTUATORS = ActuatorCfg(
  joint_names_expr=[
    # Neck
    "neck_yaw_drive",
    "neck_pitch_act",
    # Waist
    "waist_act_1",
    "waist_act_2",
    # Left leg
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw_drive",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    # Right leg
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw_drive",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # Left arm
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw_drive",
    "left_elbow_roll",
    "left_elbow_yaw_drive",
    "left_wrist_pitch_drive",
    "left_wrist_roll",
    # Right arm
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw_drive",
    "right_elbow_roll",
    "right_elbow_yaw_drive",
    "right_wrist_pitch_drive",
    "right_wrist_roll",
  ],
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
'''
    
    # Find where to insert the actuator config (before TODDLERBOT_ROBOT_CFG)
    import re
    
    # Insert before TODDLERBOT_ROBOT_CFG
    robot_cfg_pattern = r'(TODDLERBOT_ROBOT_CFG = EntityCfg\()'
    if re.search(robot_cfg_pattern, content):
        content = re.sub(
            robot_cfg_pattern,
            new_actuator_config + r'\n\1',
            content,
            count=1
        )
        
        # Update TODDLERBOT_ROBOT_CFG to include articulation
        if 'articulation=' not in content:
            content = content.replace(
                'TODDLERBOT_ROBOT_CFG = EntityCfg(\n  init_state=HOME_KEYFRAME,\n  collisions=',
                'TODDLERBOT_ROBOT_CFG = EntityCfg(\n  init_state=HOME_KEYFRAME,\n  collisions='
            )
            content = content.replace(
                '  spec_fn=get_spec,\n)',
                '  spec_fn=get_spec,\n  articulation=TODDLERBOT_ARTICULATION,\n)',
                1  # Only first occurrence
            )
    
    # Write back
    with open(constants_file, 'w') as f:
        f.write(content)
    
    print("✅ Constants file updated with actuator configuration")


def main():
    """Main conversion function."""
    print("🤖 ToddlerBot XML Converter for mjlab Framework")
    print("=" * 50)
    
    try:
        # Convert XML
        if convert_toddlerbot_xml():
            # Update constants
            create_compatible_constants()
            
            print("\n🎉 Conversion completed successfully!")
            print("\n📝 Summary of changes:")
            print("  ✓ Removed XML actuators (now using programmatic ActuatorCfg)")
            print("  ✓ Commented out explicit contact pairs (fixes MJWarp segfault)")
            print("  ✓ Preserved equality constraints (needed for gear mechanisms)")
            print("\n⚠️  Important notes:")
            print("  • Use FULL_COLLISION_WITHOUT_SELF (not FULL_COLLISION)")
            print("  • FULL_COLLISION causes MJWarp segfault due to ToddlerBot's complexity")
            print("  • Recommended max envs: 512 (1024+ may cause OOM)")
            print("\nNext steps:")
            print("1. Test with small number of environments:")
            print("   MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-ToddlerBot-2xc --env.scene.num-envs 64")
            print("2. If it works, scale up:")
            print("   MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-ToddlerBot-2xc --env.scene.num-envs 512")
            print("3. To revert changes, restore from backup:")
            print("   cp src/mjlab/asset_zoo/robots/toddlerbot_2xc/xmls/toddlerbot_2xc_original.xml \\")
            print("      src/mjlab/asset_zoo/robots/toddlerbot_2xc/xmls/toddlerbot_2xc.xml")
            
        else:
            print("❌ Conversion failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



