#!/usr/bin/env python3
"""Show ToddlerBot's default pose using the proper environment initialization.

Usage:
  python show_toddlerbot_default_pose.py                    # Default standing pose
  python show_toddlerbot_default_pose.py --pose standing    # Standing pose
  python show_toddlerbot_default_pose.py --pose walking     # Walking pose
  python show_toddlerbot_default_pose.py --pose running     # Running pose
"""

import argparse
from typing import Literal

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

# Import to register the ToddlerBot tasks
import mjlab.tasks  # noqa: F401


PoseType = Literal["standing", "walking", "running"]


class HoldPosePolicy:
  """A policy that outputs zero actions to maintain the default pose.

  With PD control, zero actions means "stay at default joint positions".
  """

  def __init__(self, num_actions: int, device: str):
    self.num_actions = num_actions
    self.device = device
    # Zero actions = hold default pose with PD control
    self.action = torch.zeros(1, num_actions, device=device)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    """Return zero actions to hold default pose with PD control."""
    batch_size = obs.shape[0]
    if batch_size == 1:
      return self.action
    return self.action.expand(batch_size, -1)


def show_default_pose(pose: PoseType = "standing"):
  """Show ToddlerBot in its default pose using viser.

  Args:
    pose: Which pose to show - "standing", "walking", or "running".
          These correspond to the different pose standards used in the
          variable_posture reward function.
  """

  configure_torch_backends()

  # Use the flat velocity task for visualization
  task = "Mjlab-Velocity-Flat-ToddlerBot-2xc"
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  print(f"[INFO] Loading task: {task}")
  print(f"[INFO] Using device: {device}")
  print(f"[INFO] Pose mode: {pose}")

  # Load environment configuration (play mode for infinite episode length)
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)

  # Use only 1 environment for visualization
  env_cfg.scene.num_envs = 1

  # Configure the velocity command based on pose type to trigger
  # the corresponding pose reward behavior
  if env_cfg.commands is not None and "twist" in env_cfg.commands:
    twist_cmd = env_cfg.commands["twist"]
    if pose == "standing":
      # Standing: very low/zero velocity command
      twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
      twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
      twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
      twist_cmd.rel_standing_envs = 1.0  # All standing
    elif pose == "walking":
      # Walking: moderate velocity (between 0.05 and 1.5)
      twist_cmd.ranges.lin_vel_x = (0.3, 0.3)
      twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
      twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
      twist_cmd.rel_standing_envs = 0.0
    elif pose == "running":
      # Running: high velocity (> 1.5)
      twist_cmd.ranges.lin_vel_x = (2.0, 2.0)
      twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
      twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
      twist_cmd.rel_standing_envs = 0.0

  print("[INFO] Creating environment...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  # Wrap for compatibility with viewer
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  num_actions = wrapped_env.num_actions

  print(f"[INFO] Environment created with {num_actions} actions")

  # Create a policy that outputs zero actions (maintains default pose with PD control)
  policy = HoldPosePolicy(num_actions, device)

  print(f"\n✅ Environment initialized to {pose} pose")
  print("   Starting viser visualization...")
  print("   Open http://localhost:8080 in your browser")
  print(f"   The robot will maintain its default {pose} pose")
  print("   Press Ctrl+C to exit\n")

  # Launch viser viewer
  viewer = ViserPlayViewer(wrapped_env, policy)
  viewer.run()

  env.close()


def main():
  parser = argparse.ArgumentParser(
    description="Show ToddlerBot's default pose using viser visualization.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Pose types:
  standing  - Tight pose constraints, for when robot is stationary (default)
  walking   - Moderate pose constraints, for walking speeds
  running   - Loose pose constraints, for running speeds

The pose types correspond to the std_standing, std_walking, and std_running
parameters in the variable_posture reward function.
    """,
  )
  parser.add_argument(
    "--pose",
    type=str,
    choices=["standing", "walking", "running"],
    default="standing",
    help="Which pose to display (default: standing)",
  )
  args = parser.parse_args()

  show_default_pose(pose=args.pose)


if __name__ == "__main__":
  main()
