from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class ToddlerBotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO configuration for ToddlerBot matching original codebase hyperparameters."""
  
  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=0.5,  # From old config (was 1.0)
      noise_std_type="log",  # From old config (was "scalar")
      actor_obs_normalization=False,  # Using custom obs scaling instead
      critic_obs_normalization=False,  # Using custom obs scaling instead
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=5e-4,  # From old config (was 0.01)
      num_learning_epochs=4,  # From old config: num_updates_per_batch (was 5)
      num_mini_batches=8,  # From old config (was 4)
      learning_rate=3e-5,  # From old config (was 1e-3)
      schedule="adaptive",
      gamma=0.97,  # From old config (was 0.99)
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )
  experiment_name: str = "toddlerbot_velocity"
  save_interval: int = 50
  num_steps_per_env: int = 20  # From old config: unroll_length (was 24)
  max_iterations: int = 30_000
