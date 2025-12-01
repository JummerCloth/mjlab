"""ToddlerBot rough terrain velocity environment configuration."""

from mjlab.asset_zoo.robots import (
  TODDLERBOT_ACTION_SCALE,
  get_toddlerbot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def toddlerbot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ToddlerBot rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_toddlerbot_robot_cfg()}

  # Override observations to use direct entity data instead of IMU sensors
  # (ToddlerBot XML doesn't have IMU sensors defined)
  cfg.observations["policy"].terms["base_lin_vel"] = ObservationTermCfg(
    func=env_mdp.base_lin_vel,
    noise=Unoise(n_min=-0.1, n_max=0.1),
  )
  cfg.observations["policy"].terms["base_ang_vel"] = ObservationTermCfg(
    func=env_mdp.base_ang_vel,
    noise=Unoise(n_min=-0.2, n_max=0.2),
  )
  cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
    func=env_mdp.base_lin_vel,
    noise=Unoise(n_min=-0.5, n_max=0.5),
  )
  cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
    func=env_mdp.base_ang_vel,
    noise=Unoise(n_min=-0.2, n_max=0.2),
  )

  # ToddlerBot uses different site names than G1
  site_names = ("left_foot_center", "right_foot_center")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TODDLERBOT_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.75

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Disable push robot for ToddlerBot
  cfg.events.pop("push_robot", None)

  # Configure pose reward std for ToddlerBot joints (including mechanism joints)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body (actuated).
    r"^(left|right)_hip_pitch$": 0.3,
    r"^(left|right)_hip_roll$": 0.15,
    r"^(left|right)_hip_yaw_drive$": 0.15,
    r"^(left|right)_knee$": 0.35,
    r"^(left|right)_ankle_pitch$": 0.25,
    r"^(left|right)_ankle_roll$": 0.1,
    # Lower body (driven - mechanism joints).
    r"^(left|right)_hip_yaw_driven$": 0.15,
    # Waist (actuated).
    r"^waist_act_[12]$": 0.15,
    # Waist (driven - mechanism joints).
    r"^waist_yaw$": 0.15,
    r"^waist_roll$": 0.08,
    # Arms (actuated).
    r"^(left|right)_shoulder_pitch$": 0.35,
    r"^(left|right)_shoulder_roll$": 0.15,
    r"^(left|right)_shoulder_yaw_drive$": 0.1,
    r"^(left|right)_elbow_roll$": 0.25,
    r"^(left|right)_elbow_yaw_drive$": 0.25,
    r"^(left|right)_wrist_pitch_drive$": 0.3,
    r"^(left|right)_wrist_roll$": 0.3,
    # Arms (driven - mechanism joints).
    r"^(left|right)_shoulder_yaw_driven$": 0.1,
    r"^(left|right)_elbow_yaw_driven$": 0.25,
    r"^(left|right)_wrist_pitch_driven$": 0.3,
    # Neck (actuated).
    r"^neck_yaw_drive$": 0.2,
    r"^neck_pitch_act$": 0.2,
    # Neck (driven - mechanism joints).
    r"^neck_yaw_driven$": 0.2,
    r"^neck_pitch$": 0.2,
    r"^neck_pitch_front$": 0.2,
    r"^neck_pitch_back$": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body (actuated).
    r"^(left|right)_hip_pitch$": 0.5,
    r"^(left|right)_hip_roll$": 0.2,
    r"^(left|right)_hip_yaw_drive$": 0.2,
    r"^(left|right)_knee$": 0.6,
    r"^(left|right)_ankle_pitch$": 0.35,
    r"^(left|right)_ankle_roll$": 0.15,
    # Lower body (driven - mechanism joints).
    r"^(left|right)_hip_yaw_driven$": 0.2,
    # Waist (actuated).
    r"^waist_act_[12]$": 0.2,
    # Waist (driven - mechanism joints).
    r"^waist_yaw$": 0.3,
    r"^waist_roll$": 0.08,
    # Arms (actuated).
    r"^(left|right)_shoulder_pitch$": 0.5,
    r"^(left|right)_shoulder_roll$": 0.2,
    r"^(left|right)_shoulder_yaw_drive$": 0.15,
    r"^(left|right)_elbow_roll$": 0.35,
    r"^(left|right)_elbow_yaw_drive$": 0.35,
    r"^(left|right)_wrist_pitch_drive$": 0.3,
    r"^(left|right)_wrist_roll$": 0.3,
    # Arms (driven - mechanism joints).
    r"^(left|right)_shoulder_yaw_driven$": 0.15,
    r"^(left|right)_elbow_yaw_driven$": 0.35,
    r"^(left|right)_wrist_pitch_driven$": 0.3,
    # Neck (actuated).
    r"^neck_yaw_drive$": 0.2,
    r"^neck_pitch_act$": 0.2,
    # Neck (driven - mechanism joints).
    r"^neck_yaw_driven$": 0.2,
    r"^neck_pitch$": 0.2,
    r"^neck_pitch_front$": 0.2,
    r"^neck_pitch_back$": 0.2,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)

  # Disable angular momentum reward (ToddlerBot XML doesn't have subtreeangmom sensor)
  cfg.rewards["angular_momentum"].weight = 0.0

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Configure command velocity curriculum for ToddlerBot (slower speeds than G1)
  # ToddlerBot is smaller, so we use more conservative velocity ranges
  if cfg.curriculum is not None and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      # Start with very conservative velocities
      {"step": 0, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.05, 0.05), "ang_vel_z": (-0.3, 0.3)},
      # Gradually increase after learning basics
      {"step": 500 * 24, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.5, 0.5)},
      # Further increase for more dynamic walking
      {"step": 5000 * 24, "lin_vel_x": (-0.4, 0.4), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-0.7, 0.7)},
    ]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
