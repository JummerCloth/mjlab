from dataclasses import dataclass, replace, field

from mjlab.asset_zoo.robots.toddlerbot_2xc.toddlerbot_2xc_constants import (
  TODDLERBOT_ACTION_SCALE,
  TODDLERBOT_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.envs import mdp
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import term
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.sim import SimulationCfg, MujocoCfg


# Custom simulation configuration for ToddlerBot with higher limits
# With FULL_COLLISION_WITHOUT_SELF, we have:
# - 38 collision geoms per robot (24 boxes + 14 foot capsules)
# - Box collisions are faster than mesh, capsules are fastest
# - 14 foot capsules generate primary ground contacts (~28 contacts/robot in normal walking)
# - Additional contacts from legs/body when falling (~40 contacts/robot max)
# For 4096 envs: 4096 × 40 = 163,840 contacts (use 170k for margin)
TODDLERBOT_SIM_CFG = SimulationCfg(
  nconmax=170_000,  # 40 contacts/env * 4096 envs with margin
  njmax=200_000,    # 45 joints * 4096 envs = 184,320
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)

# Custom observation configuration with manual scaling (from old MJX codebase)
# Instead of using automatic normalization, we manually scale observations
@dataclass
class ToddlerBotObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    """Policy observations with custom scaling."""
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel_scaled,
      noise=Unoise(n_min=-0.2, n_max=0.2),
      params={"scale": 0.25},  # Scale down angular velocity (typ. ~4 rad/s -> ~1)
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity_scaled,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      params={"scale": 1.0},  # Gravity already normalized to ~1
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel_scaled,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"scale": 1.0},  # Joint positions in radians are already reasonable
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel_scaled,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      params={"scale": 0.05},  # Scale down joint velocities (typ. ~20 rad/s -> ~1)
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "twist"}
    )

    def __post_init__(self):
      self.enable_corruption = True
      self.concatenate_terms = True
      self.concatenate_dim = -1

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    """Critic observations with custom scaling (no noise)."""
    base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel_scaled,
      params={"scale": 0.25},
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity_scaled,
      params={"scale": 1.0},
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel_scaled,
      params={"scale": 1.0},
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel_scaled,
      params={"scale": 0.05},
    )
    
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class ToddlerBotRoughEnvCfg(LocomotionVelocityEnvCfg):
  # Override simulation and observation configurations
  '''
  sim: SimulationCfg = field(default_factory=lambda: TODDLERBOT_SIM_CFG)
  observations: ToddlerBotObservationCfg = field(default_factory=ToddlerBotObservationCfg)
  '''
  
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    toddlerbot_cfg = replace(TODDLERBOT_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
    self.scene.entities = {"robot": toddlerbot_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    # Update to use the new foot capsule collision geoms (7 capsules per foot)
    geom_names = [
      "left_foot1_collision", "left_foot2_collision", "left_foot3_collision",
      "left_foot4_collision", "left_foot5_collision", "left_foot6_collision", "left_foot7_collision",
      "right_foot1_collision", "right_foot2_collision", "right_foot3_collision",
      "right_foot4_collision", "right_foot5_collision", "right_foot6_collision", "right_foot7_collision",
    ]

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.actions.joint_pos.scale = TODDLERBOT_ACTION_SCALE

    self.events.push_robot = None

    self.rewards.air_time.params["sensor_names"] = sensor_names
    
    # Configure pose reward std for all joints (including mechanism joints)
    self.rewards.pose.params["std"] = {
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

    self.viewer.body_name = "torso"
    self.commands.twist.viz.z_offset = 0.75

    self.curriculum.command_vel = None


@dataclass
class ToddlerBotRoughEnvCfg_PLAY(ToddlerBotRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
