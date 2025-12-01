from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_env_cfg import toddlerbot_flat_env_cfg
from .rl_cfg import toddlerbot_ppo_runner_cfg
from .rough_env_cfg import toddlerbot_rough_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-ToddlerBot-2xc",
  env_cfg=toddlerbot_rough_env_cfg(),
  play_env_cfg=toddlerbot_rough_env_cfg(play=True),
  rl_cfg=toddlerbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-ToddlerBot-2xc",
  env_cfg=toddlerbot_flat_env_cfg(),
  play_env_cfg=toddlerbot_flat_env_cfg(play=True),
  rl_cfg=toddlerbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
