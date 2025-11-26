# Observation Space Breakdown: G1 and ToddlerBot

## Summary

| Robot | Total Actuators | Policy Obs Dim | Critic Obs Dim | Actor Input | Critic Input |
|-------|----------------|----------------|----------------|-------------|--------------|
| **Unitree G1** | 29 joints | **71** | **71** | 71 | 142 (71+71) |
| **ToddlerBot** | 30 joints | **72** | **72** | 72 | 144 (72+72) |

---

## Unitree G1 Observation Space

### Joint Configuration

**Total: 29 Actuated Joints**

**Legs (12 joints):**
- Left leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- Right leg (6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll

**Waist (3 joints):**
- waist_yaw, waist_pitch, waist_roll

**Arms (14 joints):**
- Left arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_yaw, wrist_roll
- Right arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_yaw, wrist_roll

---

### Policy Observation (Actor Network)

**Total Dimension: 71**

```python
ObservationCfg.PolicyCfg:
    base_lin_vel       # [num_envs, 3]  - Linear velocity in body frame (x, y, z)
    base_ang_vel       # [num_envs, 3]  - Angular velocity in body frame (roll, pitch, yaw rates)
    projected_gravity  # [num_envs, 3]  - Gravity vector projected to body frame
    joint_pos          # [num_envs, 29] - Joint positions relative to default
    joint_vel          # [num_envs, 29] - Joint velocities
    actions            # [num_envs, 29] - Previous action (for temporal consistency)
    command            # [num_envs, 3]  - Velocity command (lin_vel_x, lin_vel_y, ang_vel_z)
```

**Detailed Breakdown:**
| Component | Elements | Shape | Noise | Description |
|-----------|----------|-------|-------|-------------|
| `base_lin_vel` | 3 | [N, 3] | Uniform(-0.1, 0.1) | Linear velocity: vx, vy, vz in body frame |
| `base_ang_vel` | 3 | [N, 3] | Uniform(-0.2, 0.2) | Angular velocity: ωx, ωy, ωz in body frame |
| `projected_gravity` | 3 | [N, 3] | Uniform(-0.05, 0.05) | Gravity direction in body frame |
| `joint_pos` | 29 | [N, 29] | Uniform(-0.01, 0.01) | Joint positions - default_positions |
| `joint_vel` | 29 | [N, 29] | Uniform(-1.5, 1.5) | Joint velocities in rad/s |
| `actions` | 29 | [N, 29] | None | Previous action sent to robot |
| `command` | 3 | [N, 3] | None | Target velocities: [lin_x, lin_y, ang_z] |
| **TOTAL** | **71** | [N, 71] | - | **Concatenated policy observation** |

**Noise Configuration:**
- Policy observations have noise enabled (`enable_corruption = True`)
- Each observation term can have independent noise
- Noise is applied BEFORE concatenation

---

### Critic Observation (Privileged Information)

**Total Dimension: 71 (same components as policy, but no noise)**

```python
ObservationCfg.PrivilegedCfg:
    base_lin_vel       # [num_envs, 3]  - No noise
    base_ang_vel       # [num_envs, 3]  - No noise
    projected_gravity  # [num_envs, 3]  - No noise
    joint_pos          # [num_envs, 29] - No noise
    joint_vel          # [num_envs, 29] - No noise
    actions            # [num_envs, 29] - No noise
    command            # [num_envs, 3]  - No noise
```

**Key Difference from Policy:**
- Same components but `enable_corruption = False`
- Provides ground-truth state information to critic
- Helps stabilize value function learning

---

### Network Inputs (RSL-RL)

**Actor Network Input: [num_envs, 71]**
```python
actor_input = obs["policy"]  # Shape: [num_envs, 71]
```

**Critic Network Input: [num_envs, 142]**
```python
critic_input = torch.cat([obs["policy"], obs["critic"]], dim=-1)
# Shape: [num_envs, 142] = policy (71) + critic (71)
```

**Configuration:**
```python
obs_groups = {
    "policy": ["policy"],           # Actor uses only policy obs
    "critic": ["policy", "critic"]  # Critic uses both (asymmetric actor-critic)
}
```

---

## ToddlerBot Observation Space

### Joint Configuration

**Total: 30 Actuated Joints**

**Neck (2 joints):**
- neck_yaw_drive, neck_pitch_act

**Waist (2 joints):**
- waist_act_1, waist_act_2

**Legs (12 joints):**
- Left leg (6): hip_pitch, hip_roll, hip_yaw_drive, knee, ankle_pitch, ankle_roll
- Right leg (6): hip_pitch, hip_roll, hip_yaw_drive, knee, ankle_pitch, ankle_roll

**Arms (14 joints):**
- Left arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw_drive, elbow_roll, elbow_yaw_drive, wrist_pitch_drive, wrist_roll
- Right arm (7): shoulder_pitch, shoulder_roll, shoulder_yaw_drive, elbow_roll, elbow_yaw_drive, wrist_pitch_drive, wrist_roll

**Note:** ToddlerBot has mechanism/driven joints (e.g., `*_driven`, `*_yaw_driven`) that are NOT actuated - these are passive joints in the mechanism and not included in the observation/action space.

---

### Policy Observation (Actor Network)

**Total Dimension: 72**

```python
ObservationCfg.PolicyCfg:
    base_lin_vel       # [num_envs, 3]  - Linear velocity in body frame
    base_ang_vel       # [num_envs, 3]  - Angular velocity in body frame
    projected_gravity  # [num_envs, 3]  - Gravity vector projected to body frame
    joint_pos          # [num_envs, 30] - Joint positions relative to default
    joint_vel          # [num_envs, 30] - Joint velocities
    actions            # [num_envs, 30] - Previous action
    command            # [num_envs, 3]  - Velocity command
```

**Detailed Breakdown:**
| Component | Elements | Shape | Noise | Description |
|-----------|----------|-------|-------|-------------|
| `base_lin_vel` | 3 | [N, 3] | Uniform(-0.1, 0.1) | Linear velocity: vx, vy, vz in body frame |
| `base_ang_vel` | 3 | [N, 3] | Uniform(-0.2, 0.2) | Angular velocity: ωx, ωy, ωz in body frame |
| `projected_gravity` | 3 | [N, 3] | Uniform(-0.05, 0.05) | Gravity direction in body frame |
| `joint_pos` | 30 | [N, 30] | Uniform(-0.01, 0.01) | Joint positions - default_positions |
| `joint_vel` | 30 | [N, 30] | Uniform(-1.5, 1.5) | Joint velocities in rad/s |
| `actions` | 30 | [N, 30] | None | Previous action sent to robot |
| `command` | 3 | [N, 3] | None | Target velocities: [lin_x, lin_y, ang_z] |
| **TOTAL** | **72** | [N, 72] | - | **Concatenated policy observation** |

---

### Critic Observation (Privileged Information)

**Total Dimension: 72 (same as policy, no noise)**

---

### Network Inputs (RSL-RL)

**Actor Network Input: [num_envs, 72]**
```python
actor_input = obs["policy"]  # Shape: [num_envs, 72]
```

**Critic Network Input: [num_envs, 144]**
```python
critic_input = torch.cat([obs["policy"], obs["critic"]], dim=-1)
# Shape: [num_envs, 144] = policy (72) + critic (72)
```

---

## Observation Space Comparison: G1 vs ToddlerBot

| Component | G1 | ToddlerBot | Difference |
|-----------|----|-----------| -----------|
| Base linear velocity | 3 | 3 | Same |
| Base angular velocity | 3 | 3 | Same |
| Projected gravity | 3 | 3 | Same |
| Joint positions | 29 | 30 | ToddlerBot has 1 more joint |
| Joint velocities | 29 | 30 | ToddlerBot has 1 more joint |
| Last actions | 29 | 30 | ToddlerBot has 1 more joint |
| Commands | 3 | 3 | Same |
| **Total Policy Obs** | **71** | **72** | **+1 for ToddlerBot** |
| **Total Critic Obs** | **71** | **72** | **+1 for ToddlerBot** |
| **Actor Input** | **71** | **72** | **+1 for ToddlerBot** |
| **Critic Input** | **142** | **144** | **+2 for ToddlerBot** |

---

## Key Observations

### Observation Structure
1. **Grouped observations**: Policy and critic are separate groups
2. **Concatenated within groups**: All terms within a group are concatenated into a single tensor
3. **Asymmetric actor-critic**: Critic sees both policy and critic observations (privileged info)

### Noise Application
- **Policy observations**: Noisy (simulates sensor noise for sim-to-real transfer)
- **Critic observations**: Clean ground truth (helps with learning stability)
- **Noise is per-term**: Each observation component can have different noise parameters

### Missing from MJX
- **Frame stacking**: MJX stacks 15 timesteps of observations
  - If mjlab had frame stacking of 15 frames:
    - G1 policy obs would be: 71 × 15 = **1,065 dimensions**
    - ToddlerBot policy obs would be: 72 × 15 = **1,080 dimensions**

### Command Space
Both robots use the same 3D command space:
- `lin_vel_x`: Forward/backward velocity (m/s)
- `lin_vel_y`: Left/right velocity (m/s)  
- `ang_vel_z`: Turning rate (rad/s)

---

## Example: Accessing Observations in Code

### Environment Step
```python
# Create environment
env = create_env(cfg)
obs_dict, _ = env.reset()

# Observation dictionary structure
print(obs_dict.keys())  # ['policy', 'critic']
print(obs_dict['policy'].shape)  # torch.Size([4096, 71]) for G1
print(obs_dict['critic'].shape)  # torch.Size([4096, 71]) for G1
```

### In Training Loop
```python
# Wrapped as TensorDict for RSL-RL
obs = TensorDict({
    'policy': torch.Tensor([4096, 71]),
    'critic': torch.Tensor([4096, 71])
}, batch_size=[4096])

# RSL-RL extracts observations
actor_obs = obs['policy']                              # [4096, 71]
critic_obs = torch.cat([obs['policy'], obs['critic']], dim=-1)  # [4096, 142]

# Forward pass
actions = policy.act(obs)  # Uses actor_obs internally
value = policy.evaluate(obs, actions)  # Uses critic_obs internally
```

### Inspecting Individual Terms
```python
# Get individual observation terms (before concatenation)
obs_manager = env.observation_manager

# See what's in policy group
print(obs_manager._group_obs_term_names['policy'])
# Output: ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 
#          'joint_pos', 'joint_vel', 'actions', 'command']

# Get dimensions of each term
print(obs_manager._group_obs_term_dim['policy'])
# Output: [(3,), (3,), (3,), (29,), (29,), (29,), (3,)] for G1
```

---

## Configuration Files Reference

**G1:** `src/mjlab/tasks/velocity/config/g1/rough_env_cfg.py`
**ToddlerBot:** `src/mjlab/tasks/velocity/config/toddlerbot_2xc/rough_env_cfg.py`
**Base Config:** `src/mjlab/tasks/velocity/velocity_env_cfg.py`

---

## Notes for Migration from MJX

### Major Differences:
1. **No frame stacking** in current mjlab (MJX uses 15 frames)
2. **Observation grouping** is more explicit in mjlab (policy/critic groups)
3. **Noise configuration** is more modular in mjlab (per-term config)
4. **Observation scaling** in mjlab uses manual scaling functions instead of automatic normalization

### To match MJX behavior:
- Implement frame stacking to concatenate last 15 observations
- This would multiply observation dimensions by 15
- Need to maintain rolling buffer in ObservationManager

### Advantages of mjlab approach:
- Clearer separation of policy vs critic observations
- More flexible noise configuration per observation term
- Easier to debug individual observation components
- Better support for asymmetric actor-critic architectures

