#!/usr/bin/env python3
"""Show ToddlerBot's default pose using the proper environment initialization."""

import gymnasium as gym
import torch
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserViewer

# Import to register the ToddlerBot tasks
import mjlab.tasks.velocity.config.toddlerbot_2xc  # noqa: F401


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


def show_default_pose():
    """Show ToddlerBot in its default pose using viser."""
    
    configure_torch_backends()
    
    # Use the same task configuration as play
    task = "Mjlab-Velocity-Flat-ToddlerBot-2xc"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Loading task: {task}")
    print(f"[INFO] Using device: {device}")
    
    # Load environment and agent configuration
    env_cfg = cast(
        ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
    )
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
    )
    
    # Use only 1 environment for visualization
    env_cfg.scene.num_envs = 1
    
    print("[INFO] Creating environment...")
    env = gym.make(task, cfg=env_cfg, device=device, render_mode=None)
    
    # Get the number of actions from the environment
    from mjlab.rl import RslRlVecEnvWrapper
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    num_actions = wrapped_env.num_actions
    
    print(f"[INFO] Environment created with {num_actions} actions")
    
    # Create a policy that outputs zero actions (maintains default pose with PD control)
    policy = HoldPosePolicy(num_actions, device)
    
    print("\n✅ Environment initialized to default pose")
    print("   Starting viser visualization...")
    print("   Open http://localhost:8080 in your browser")
    print("   The robot will maintain its default standing pose")
    print("   Press Ctrl+C to exit\n")
    
    # Launch viser viewer
    viewer = ViserViewer(wrapped_env, policy, render_all_envs=True)
    viewer.run()
    
    env.close()


if __name__ == "__main__":
    show_default_pose()

