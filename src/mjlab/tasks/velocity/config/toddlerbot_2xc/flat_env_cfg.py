from dataclasses import dataclass

from mjlab.tasks.velocity.config.toddlerbot_2xc.rough_env_cfg import (
  ToddlerBotRoughEnvCfg,
)


@dataclass
class ToddlerBotFlatEnvCfg(ToddlerBotRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None

    self.curriculum.command_vel = None

    # assert self.events.push_robot is not None
    # self.events.push_robot.params["velocity_range"] = {
    #   "x": (-0.05, 0.05),
    #   "y": (-0.05, 0.05),
    # }


@dataclass
class ToddlerBotFlatEnvCfg_PLAY(ToddlerBotFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
    
    # Increase command ranges for more visible movement during eval
    self.commands.twist.ranges.lin_vel_x = (-0.5, 0.5)  # 2x faster forward/back
    self.commands.twist.ranges.lin_vel_y = (-0.2, 0.2)  # 2x faster lateral
    self.commands.twist.ranges.ang_vel_z = (-2.0, 2.0)  # 2x faster turning
    self.commands.twist.rel_standing_envs = 0.0  # No standing, always moving!