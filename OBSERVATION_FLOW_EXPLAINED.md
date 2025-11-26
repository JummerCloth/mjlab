# mjlab Observation Flow: Complete Explanation

## Overview

This document explains how observations are computed, structured, and fed into the policy network in mjlab.

---

## High-Level Flow

```
Simulation State
    ↓
ObservationManager.compute()
    ↓
Dictionary of Observation Groups
    ↓
RslRlVecEnvWrapper (wraps as TensorDict)
    ↓
RSL-RL OnPolicyRunner
    ↓
ActorCritic Policy Network
```

---

## Step-by-Step Breakdown

### 1. Observation Computation in Environment

**Location**: `src/mjlab/managers/observation_manager.py`

```python
class ObservationManager(ManagerBase):
    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name)
        self._obs_buffer = obs_buffer
        return obs_buffer
```

**Key Points**:
- Returns a **dictionary** with observation groups as keys
- Each group can be a **single tensor** (if concatenated) or **dict of tensors**
- Groups are defined in your config (e.g., `policy`, `critic`)

---

### 2. Observation Group Structure

**Example Config**: `src/mjlab/tasks/velocity/velocity_env_cfg.py`

```python
@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for the actor network."""
        base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)
        base_ang_vel: ObsTerm = term(ObsTerm, func=mdp.base_ang_vel)
        projected_gravity: ObsTerm = term(ObsTerm, func=mdp.projected_gravity)
        joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)
        joint_vel: ObsTerm = term(ObsTerm, func=mdp.joint_vel_rel)
        actions: ObsTerm = term(ObsTerm, func=mdp.last_action)
        command: ObsTerm = term(ObsTerm, func=mdp.generated_commands, ...)
        
        def __post_init__(self):
            self.enable_corruption = True  # Add noise
            self.concatenate_terms = True  # Concat into single tensor
            self.concatenate_dim = -1      # Concat along last dim
    
    @dataclass  
    class PrivilegedCfg(PolicyCfg):
        """Privileged observations for critic network (no noise)."""
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False  # No noise for critic
    
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)
```

**Resulting Structure**:
```python
obs_dict = {
    "policy": torch.Tensor([num_envs, obs_dim]),  # Concatenated
    "critic": torch.Tensor([num_envs, privileged_obs_dim])  # Concatenated
}
```

---

### 3. Observation Group Computation

**Location**: `src/mjlab/managers/observation_manager.py`

```python
def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
    group_term_names = self._group_obs_term_names[group_name]
    group_obs: dict[str, torch.Tensor] = {}
    
    # Compute each observation term
    for term_name, term_cfg in zip(group_term_names, self._group_obs_term_cfgs[group_name]):
        obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
        
        # Apply noise (only if enable_corruption=True)
        if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
            obs = term_cfg.noise.apply(obs)
        elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
            obs = self._group_obs_class_instances[term_name](obs)
        
        group_obs[term_name] = obs
    
    # Concatenate if configured
    if self._group_obs_concatenate[group_name]:
        return torch.cat(
            list(group_obs.values()), 
            dim=self._group_obs_concatenate_dim[group_name]
        )
    return group_obs
```

**What This Does**:
1. For each observation term (e.g., `base_lin_vel`, `joint_pos`):
   - Call the observation function (e.g., `mdp.base_lin_vel(env)`)
   - Apply noise if configured
2. If `concatenate_terms=True`, concatenate all into single tensor
3. Otherwise, return dictionary of individual tensors

---

### 4. Observation Terms (Functions)

**Location**: `src/mjlab/envs/mdp/observations.py`

```python
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Returns base linear velocity in body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b  # Shape: [num_envs, 3]

def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Returns joint positions relative to default."""
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]
    # Shape: [num_envs, num_joints]

def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """Returns previous action."""
    if action_name is None:
        return env.action_manager.action  # Shape: [num_envs, action_dim]
    return env.action_manager.get_term(action_name).raw_action
```

**Each function returns**: `torch.Tensor` with shape `[num_envs, ...]`

---

### 5. Environment Step Return

**Location**: `src/mjlab/envs/manager_based_rl_env.py`

```python
def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    # ... simulation steps ...
    
    # Compute observations after simulation
    self.obs_buf = self.observation_manager.compute()
    
    return (
        self.obs_buf,      # dict[str, torch.Tensor]
        self.reward_buf,   # torch.Tensor
        self.reset_terminated,  # torch.Tensor (bool)
        self.reset_time_outs,   # torch.Tensor (bool)
        self.extras,       # dict
    )
```

---

### 6. RSL-RL Wrapper (TensorDict Conversion)

**Location**: `src/mjlab/rl/vecenv_wrapper.py`

```python
class RslRlVecEnvWrapper(VecEnv):
    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs])
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # ...
        return (
            TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]),
            rew,
            dones,
            extras,
        )
```

**TensorDict Structure**:
```python
TensorDict({
    "policy": Tensor([num_envs, policy_obs_dim]),
    "critic": Tensor([num_envs, critic_obs_dim]),
}, batch_size=[num_envs])
```

**Why TensorDict?**
- RSL-RL uses TensorDict to handle multi-group observations
- Allows separate observations for actor and critic networks
- Efficient batching and device management

---

### 7. RSL-RL Configuration

**Location**: `src/mjlab/rl/config.py`

```python
@dataclass
class RslRlBaseRunnerCfg:
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            "policy": ["policy"],      # Actor uses "policy" group
            "critic": ["policy", "critic"]  # Critic uses both groups
        },
    )
```

**What This Means**:
- **Actor network** receives: `obs_dict["policy"]`
- **Critic network** receives: concatenation of `obs_dict["policy"]` + `obs_dict["critic"]`

**Example**:
```python
# If policy obs is [4096, 50] and critic obs is [4096, 100]:
actor_input = obs["policy"]              # [4096, 50]
critic_input = torch.cat([obs["policy"], obs["critic"]], dim=-1)  # [4096, 150]
```

---

### 8. RSL-RL Policy Network Usage

**In RSL-RL's ActorCritic class** (from rsl_rl package):

```python
class ActorCritic(nn.Module):
    def act(self, observations: TensorDict, **kwargs):
        # Extract actor observations
        if self.obs_groups["policy"]:
            actor_obs = torch.cat([observations[k] for k in self.obs_groups["policy"]], dim=-1)
        
        # Apply observation normalization if enabled
        if self.actor_obs_normalization:
            actor_obs = self.actor_obs_normalizer(actor_obs)
        
        # Forward through actor network
        return self.actor(actor_obs)
    
    def evaluate(self, observations: TensorDict, actions):
        # Extract critic observations
        if self.obs_groups["critic"]:
            critic_obs = torch.cat([observations[k] for k in self.obs_groups["critic"]], dim=-1)
        
        # Apply observation normalization if enabled
        if self.critic_obs_normalization:
            critic_obs = self.critic_obs_normalizer(critic_obs)
        
        # Forward through critic network
        return self.critic(critic_obs)
```

---

## Complete Example: Velocity Tracking Task

### Configuration

```python
@dataclass
class ObservationCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)      # 3
        base_ang_vel: ObsTerm = term(ObsTerm, func=mdp.base_ang_vel)      # 3
        projected_gravity: ObsTerm = term(ObsTerm, func=mdp.projected_gravity)  # 3
        joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)        # 23 (for G1)
        joint_vel: ObsTerm = term(ObsTerm, func=mdp.joint_vel_rel)        # 23
        actions: ObsTerm = term(ObsTerm, func=mdp.last_action)            # 23
        command: ObsTerm = term(ObsTerm, func=mdp.generated_commands)     # 3
        
        def __post_init__(self):
            self.concatenate_terms = True
    
    @dataclass
    class PrivilegedCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
    
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)
```

### Resulting Observation Dimensions

```python
# Policy observation (with noise)
policy_obs = [
    base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3) +
    joint_pos (23) + joint_vel (23) + actions (23) + command (3)
] = 81 dimensions

# Critic observation (no noise, same components)
critic_obs = 81 dimensions

# Final shapes
obs_dict = {
    "policy": torch.Tensor([4096, 81]),
    "critic": torch.Tensor([4096, 81])
}

# Fed to networks
actor_input = obs["policy"]                            # [4096, 81]
critic_input = torch.cat([obs["policy"], obs["critic"]], dim=-1)  # [4096, 162]
```

---

## Key Differences from MJX

### MJX Approach (JAX)
```python
# MJX concatenates everything first, then stacks frames
obs = jnp.concatenate([
    base_lin_vel,
    base_ang_vel,
    joint_pos,
    # ... all components
])

# Then frame stacking
obs_stacked = jnp.roll(obs_history, obs.size).at[:obs.size].set(obs)

# Returns single observation vector
return {"state": obs_stacked, "privileged_state": privileged_obs_stacked}
```

### mjlab Approach (PyTorch)
```python
# mjlab computes individual terms, optionally applies noise, then concatenates
group_obs = {
    "base_lin_vel": base_lin_vel(env),
    "base_ang_vel": base_ang_vel(env),
    # ... apply noise to each ...
}

# Concatenate if configured
if concatenate_terms:
    obs = torch.cat(list(group_obs.values()), dim=-1)

# Returns dictionary of groups (NO FRAME STACKING YET)
return {"policy": obs, "critic": privileged_obs}
```

**Missing in mjlab**: Frame stacking across time steps

---

## Observation Noise Application

### Noise Configuration

```python
@dataclass
class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
        ObsTerm,
        func=mdp.base_lin_vel,
        noise=Unoise(n_min=-0.1, n_max=0.1)  # Uniform noise
    )
    base_ang_vel: ObsTerm = term(
        ObsTerm,
        func=mdp.base_ang_vel,
        noise=GaussianNoise(mean=0.0, std=0.1)  # Gaussian noise
    )
```

### Noise Types

**1. Simple Noise** (`NoiseCfg`):
```python
class UniformNoiseCfg(NoiseCfg):
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(data) * (self.n_max - self.n_min) + self.n_min
        return data + noise
```

**2. Noise Models** (`NoiseModelCfg`):
```python
class NoiseModelWithAdditiveBias(NoiseModel):
    """Adds constant bias per episode + per-step noise."""
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        noisy_data = self._noise_cfg.apply(data)  # Per-step noise
        return noisy_data + self._bias              # + constant bias
```

**Applied in**: `ObservationManager.compute_group()`

---

## Observation Scaling

### Manual Scaling (MJX-style)

```python
@dataclass
class PolicyCfg(ObsGroup):
    base_ang_vel: ObsTerm = term(
        ObsTerm,
        func=mdp.base_ang_vel_scaled,
        params={"scale": 0.25}  # Scale down from ~4 rad/s to ~1
    )
    joint_vel: ObsTerm = term(
        ObsTerm,
        func=mdp.joint_vel_rel_scaled,
        params={"scale": 0.05}  # Scale down from ~20 rad/s to ~1
    )
```

### Automatic Normalization (RSL-RL)

```python
@dataclass
class RslRlPpoActorCriticCfg:
    actor_obs_normalization: bool = True
    # Enables running mean/std normalization in RSL-RL
```

**Best Practice**: Use manual scaling for consistent sim-to-real transfer

---

## Debugging Observations

### Check Observation Dimensions

```python
# In your environment
env = create_env(cfg)
obs, _ = env.reset()

for group_name, group_obs in obs.items():
    print(f"{group_name}: {group_obs.shape}")
    
# Output:
# policy: torch.Size([4096, 81])
# critic: torch.Size([4096, 81])
```

### Check Individual Terms

```python
env.observation_manager.compute_group("policy")
# Returns concatenated tensor

# To see individual terms:
env.observation_manager._group_obs_term_names["policy"]
# ['base_lin_vel', 'base_ang_vel', 'projected_gravity', ...]

env.observation_manager._group_obs_term_dim["policy"]  
# [(3,), (3,), (3,), (23,), ...]
```

### Verify Observation Flow

```python
# Step 1: Raw observation functions
obs_terms = {}
obs_terms["base_lin_vel"] = mdp.base_lin_vel(env)  # [4096, 3]
obs_terms["joint_pos"] = mdp.joint_pos_rel(env)    # [4096, 23]

# Step 2: After noise
obs_noisy = {}
for name, obs in obs_terms.items():
    cfg = obs_manager._group_obs_term_cfgs["policy"][idx]
    obs_noisy[name] = cfg.noise.apply(obs) if cfg.noise else obs

# Step 3: After concatenation
policy_obs = torch.cat(list(obs_noisy.values()), dim=-1)  # [4096, total_dim]

# Step 4: Wrapped in TensorDict
obs_dict = TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[4096])

# Step 5: Fed to policy
actions = policy.act(obs_dict)
```

---

## Summary

### Data Flow
1. **Environment** computes observations via `ObservationManager`
2. **Observation functions** extract data from simulation state
3. **Noise** is applied per-term (if configured)
4. **Concatenation** produces single tensor per group
5. **TensorDict** wraps groups for RSL-RL
6. **Policy network** extracts relevant groups and processes

### Key Points
- ✅ Observations are **grouped** (policy, critic, etc.)
- ✅ Each group can have **different noise** settings
- ✅ Terms are **concatenated** within each group
- ✅ Actor and critic can use **different observation groups**
- ❌ **No frame stacking** yet (unlike MJX)
- ✅ Clean separation of concerns (managers, terms, configs)

### To Match MJX Behavior
You need to implement frame stacking that:
1. Maintains a rolling buffer of past observations
2. Stacks them along the feature dimension
3. Updates the buffer each step before feeding to policy

See `MIGRATION_CHECKLIST.md` for implementation details.

