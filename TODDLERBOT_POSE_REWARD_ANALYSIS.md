# ToddlerBot Pose Reward Analysis

## What Does the Pose Reward Track?

The pose reward in ToddlerBot **tracks current joint positions to the default "home" configuration** defined in the robot's initial state.

---

## Reward Function

**Location**: `src/mjlab/envs/mdp/rewards.py`

```python
class posture:
    def __call__(self, env, std, asset_cfg):
        current_joint_pos = asset.data.joint_pos[:, joint_ids]
        desired_joint_pos = self.default_joint_pos[:, joint_ids]  # Target positions
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))
```

**Formula**: 
```
reward = exp(-mean((q_current - q_default)^2 / σ^2))
```

Where:
- `q_current` = current joint positions
- `q_default` = default joint positions (from HOME_KEYFRAME)
- `σ` = standard deviation (tolerance) per joint

**Reward Range**: [0, 1]
- **1.0** = perfect match to default pose
- **→ 0** = large deviation from default pose

---

## Target Pose: HOME_KEYFRAME

**Location**: `src/mjlab/asset_zoo/robots/toddlerbot_2xc/toddlerbot_2xc_constants.py`

```python
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.02020514, 0.0, 0.310053),  # Standing position: x, y, z(height)
    joint_pos={
        # Legs - slightly bent knees for stability
        "left_hip_pitch": -0.091312,      # -5.23°  (slight forward lean)
        "left_knee": -0.380812,           # -21.82° (bent knee)
        "left_ankle_pitch": -0.2895,      # -16.59° (ankle compensation)
        
        "right_hip_pitch": 0.091312,      # +5.23°  (opposite side)
        "right_knee": 0.380812,           # +21.82° (bent knee)
        "right_ankle_pitch": 0.2895,      # +16.59° (ankle compensation)
        
        # Arms - raised/bent position
        "left_shoulder_pitch": 0.174533,     # 10° (arm slightly forward)
        ".*_shoulder_roll": 0.087266,        # 5° (arms away from body)
        ".*_elbow_roll": -0.523599,          # -30° (elbows bent)
        
        # Complex mechanism joints (driven joints follow actuated ones)
        "left_shoulder_yaw_drive": 1.570796,   # 90° (mechanism position)
        "left_shoulder_yaw_driven": -1.570796, # -90° (coupled mechanism)
        "left_elbow_yaw_drive": -1.570796,     # -90°
        "left_elbow_yaw_driven": 1.570796,     # 90°
        "left_wrist_pitch_drive": 1.22173,     # 70°
        "left_wrist_pitch_driven": -1.22173,   # -70°
        
        # Right arm (mirrored)
        "right_shoulder_pitch": -0.174533,
        "right_shoulder_yaw_drive": -1.570796,
        "right_shoulder_yaw_driven": 1.570796,
        "right_elbow_yaw_drive": 1.570796,
        "right_elbow_yaw_driven": -1.570796,
        "right_wrist_pitch_drive": -1.22173,
        "right_wrist_pitch_driven": 1.22173,
    },
    joint_vel={".*": 0.0},
)
```

### Physical Description

**This is a stable standing pose with:**
- **Height**: ~31 cm off ground
- **Knees bent**: ~22° for shock absorption and stability
- **Arms raised**: Elbows bent at 30°, arms slightly away from body
- **Balance**: Hip and ankle angles compensate for bent knees

**Purpose**: Encourages the robot to maintain an upright, balanced posture during locomotion rather than collapsing or adopting extreme configurations.

---

## Standard Deviations (Tolerances)

**Configuration**: `src/mjlab/tasks/velocity/config/toddlerbot_2xc/rough_env_cfg.py`

The `std` parameter controls how strictly each joint is penalized for deviating from default:
- **Smaller std** = stricter tracking (larger penalty for deviation)
- **Larger std** = more tolerance (smaller penalty for deviation)

```python
self.rewards.pose.params["std"] = {
    # === LEGS (Actuated) ===
    "left/right_hip_pitch":     0.3,   # More tolerance (for walking motion)
    "left/right_hip_roll":      0.15,  # Moderate tolerance
    "left/right_hip_yaw_drive": 0.15,  # Moderate tolerance
    "left/right_knee":          0.35,  # Most tolerance (knees move a lot)
    "left/right_ankle_pitch":   0.25,  # Good tolerance
    "left/right_ankle_roll":    0.1,   # Strict (important for balance)
    
    # === LEGS (Mechanism - driven joints) ===
    "left/right_hip_yaw_driven": 0.15,
    
    # === WAIST (Actuated) ===
    "waist_act_1/2": 0.15,
    
    # === WAIST (Mechanism) ===
    "waist_yaw":  0.15,
    "waist_roll": 0.08,   # Very strict (keep torso upright)
    
    # === ARMS (Actuated) ===
    "left/right_shoulder_pitch":     0.35,  # More tolerance
    "left/right_shoulder_roll":      0.15,
    "left/right_shoulder_yaw_drive": 0.1,   # Moderate
    "left/right_elbow_roll":         0.25,
    "left/right_elbow_yaw_drive":    0.25,
    "left/right_wrist_pitch_drive":  0.3,
    "left/right_wrist_roll":         0.3,
    
    # === ARMS (Mechanism) ===
    "left/right_shoulder_yaw_driven":  0.1,
    "left/right_elbow_yaw_driven":     0.25,
    "left/right_wrist_pitch_driven":   0.3,
    
    # === NECK (Actuated) ===
    "neck_yaw_drive":  0.2,
    "neck_pitch_act":  0.2,
    
    # === NECK (Mechanism) ===
    "neck_yaw_driven":     0.2,
    "neck_pitch":          0.2,
    "neck_pitch_front":    0.2,
    "neck_pitch_back":     0.2,
}
```

### Key Insights from Tolerances

**Strictest joints (smallest std):**
- `waist_roll` (0.08): Keep torso upright and balanced
- `ankle_roll` (0.1): Critical for lateral balance
- `shoulder_yaw`, `hip_yaw` (0.1-0.15): Maintain body orientation

**Most tolerant joints (largest std):**
- `knee` (0.35): Needs to flex significantly for walking
- `shoulder_pitch` (0.35): Arms can swing during walking
- `hip_pitch` (0.3): Required for leg motion
- `wrist` (0.3): Hand position less critical

**Strategy**: The std values allow natural walking motion (knees, hips) while maintaining stability (waist, ankles).

---

## Reward Weight

**Configuration**: `src/mjlab/tasks/velocity/velocity_env_cfg.py`

```python
@dataclass
class RewardCfg:
    pose: RewardTerm = term(
        RewardTerm,
        func=mdp.posture,
        weight=1.0,  # Same weight as velocity tracking
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "std": [],  # Overridden in robot-specific config
        },
    )
```

**Weight = 1.0**: The pose reward has equal importance to velocity tracking rewards.

**Total reward** = `velocity_tracking + pose_reward + other_terms`

---

## Example Calculation

### Scenario: Robot tracking pose perfectly
```python
# All joints at default positions
current_pos = default_pos  # No error
error_squared = 0
reward = exp(-mean(0 / std^2)) = exp(0) = 1.0
```

### Scenario: Knee deviates by 0.35 rad (~20°)
```python
# Knee std = 0.35
error = 0.35 rad
error_squared = 0.1225
normalized_error = 0.1225 / (0.35^2) = 0.1225 / 0.1225 = 1.0
reward_contribution = exp(-1.0) ≈ 0.368

# If this is the only error and averaged across all joints:
mean_error = 1.0 / num_joints  # Small contribution
reward ≈ exp(-small_number) ≈ 0.95-0.99
```

### Scenario: Waist roll deviates by 0.08 rad
```python
# Waist roll std = 0.08 (very strict)
error = 0.08 rad
error_squared = 0.0064
normalized_error = 0.0064 / (0.08^2) = 0.0064 / 0.0064 = 1.0
reward_contribution = exp(-1.0) ≈ 0.368

# Same math, but waist roll is critical for balance
# Deviation is penalized equally despite smaller angle
```

---

## Why This Reward?

### Purpose
1. **Maintain upright posture** during locomotion
2. **Prevent collapse** or extreme joint configurations
3. **Encourage stable gaits** rather than chaotic motion
4. **Balance with task objectives** (velocity tracking)

### Design Philosophy
- **Soft constraint**: Not a hard penalty, allows natural motion
- **Differentiable**: Smooth gradients for learning
- **Normalized**: Each joint penalized proportionally to its importance
- **Balanced**: Works alongside velocity tracking without dominating

---

## Comparison to MJX DeepMimic

### MJX Approach (Motion Imitation)
```python
# MJX tracks reference motion frame-by-frame
reward = exp(-sigma * ||q_current - q_reference(t)||^2)

# q_reference(t) changes every timestep from motion capture
```

### mjlab Approach (Posture Regularization)
```python
# mjlab tracks static default pose
reward = exp(-mean((q_current - q_default)^2 / std^2))

# q_default is constant (HOME_KEYFRAME)
```

**Key Difference**:
- **MJX**: Tracks time-varying reference trajectory (DeepMimic style)
- **mjlab**: Regularizes toward static default configuration

**Use Case**:
- **MJX**: For replicating specific motions (dancing, acrobatics)
- **mjlab**: For general locomotion with posture stability

---

## Tuning Guide

### If robot collapses or falls:
- **Increase pose reward weight** (1.0 → 2.0)
- **Decrease std for critical joints** (knees, waist, ankles)

### If robot is too stiff, can't walk well:
- **Decrease pose reward weight** (1.0 → 0.5)
- **Increase std for leg joints** (hips, knees)

### If robot tilts/leans:
- **Decrease waist_roll std** (make stricter)
- **Check projected_gravity reward** (should also encourage upright)

### If arms flail around:
- **Decrease arm joint std** (0.35 → 0.2)
- Or **mask out arm joints** from pose reward

---

## Code to Visualize Target Pose

```python
from mjlab.asset_zoo.robots.toddlerbot_2xc.toddlerbot_2xc_constants import TODDLERBOT_ROBOT_CFG
from mjlab.entity import Entity
import mujoco.viewer as viewer

# Create robot in default pose
robot = Entity(TODDLERBOT_ROBOT_CFG)
model = robot.spec.compile()

# Launch viewer to see HOME_KEYFRAME pose
viewer.launch(model)
```

**Result**: You'll see ToddlerBot in a stable standing position with slightly bent knees and raised arms.

---

## Summary

| Aspect | Details |
|--------|---------|
| **Target** | HOME_KEYFRAME configuration (stable standing pose) |
| **Method** | Exponential penalty on joint position deviation |
| **Per-Joint Tolerance** | Configured via `std` parameter (0.08 to 0.35 rad) |
| **Strictest Joints** | waist_roll (0.08), ankle_roll (0.1) |
| **Most Tolerant** | knee (0.35), shoulder_pitch (0.35) |
| **Purpose** | Maintain stable upright posture during locomotion |
| **Weight** | 1.0 (equal to velocity tracking) |
| **Reward Range** | [0, 1] where 1 = perfect pose |

The pose reward encourages ToddlerBot to stay close to a stable standing configuration while still allowing enough flexibility for walking motions.

