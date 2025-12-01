"""ToddlerBot flat terrain velocity environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.toddlerbot_2xc.rough_env_cfg import (
  toddlerbot_rough_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def toddlerbot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ToddlerBot flat terrain velocity configuration."""
  cfg = toddlerbot_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  if "terrain_levels" in cfg.curriculum:
    del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Increase command ranges for more visible movement during eval
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-2.0, 2.0)
    twist_cmd.rel_standing_envs = 0.0  # No standing, always moving!

  return cfg
