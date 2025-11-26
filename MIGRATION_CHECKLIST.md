# MJX to mjlab Migration Checklist

This document outlines the work required to migrate training pipelines from the original MJX (JAX-based) codebase to mjlab (PyTorch + MuJoCo Warp).

## Summary

**Overall Status**: ~75% feature parity

**Major Differences**:
- **Backend**: JAX/MJX → PyTorch/MuJoCo Warp
- **Config System**: Gin files → Python dataclasses
- **Architecture**: More modular manager-based system in mjlab

---

## Critical Migration Tasks (Must Complete)

### 1. ⚠️ Frame Stacking for Observations [HIGH PRIORITY]
**Status**: MISSING in mjlab

**MJX Implementation**:
```python
# Stacks 15 frames of observation history
obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)
```

**Required for mjlab**:
- Add frame buffer to `ObservationManager`
- Implement rolling buffer for observation history
- Add `frame_stack` parameter to observation config

**Estimated Effort**: 1-2 days

---

### 2. ⚠️ Configuration System Migration [HIGH PRIORITY]
**Status**: DIFFERENT - Requires manual porting

**MJX**: Uses Gin configuration files (`.gin`)
```python
@gin.configurable
def get_config():
    return {
        'action_scale': 0.25,
        'cycle_time': 0.72,
    }
```

**mjlab**: Uses Python dataclasses
```python
@dataclass
class EnvCfg(ManagerBasedRlEnvCfg):
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    
    def __post_init__(self):
        self.actions.joint_pos.scale = 0.25
```

**Migration Steps**:
1. Identify all `.gin` config files in MJX codebase
2. Convert each config to dataclass hierarchy in mjlab
3. Map MJX parameters to mjlab equivalents (see mapping table below)
4. Test each config by comparing environment behavior

**Key Config Mappings**:
| MJX Parameter | mjlab Equivalent | Location |
|---------------|------------------|----------|
| `action_scale` | `actions.joint_pos.scale` | ActionsCfg |
| `n_frames` | `decimation` | SimulationCfg |
| `frame_stack` | NOT IMPLEMENTED | Need to add |
| `cycle_time` | N/A (motion file based) | Motion files |
| `reward_scales.*` | `rewards.*.weight` | RewardsCfg |
| `observation_scales.*` | Custom scaling in obs funcs | ObservationsCfg |

**Estimated Effort**: 2-3 days per environment

---

### 3. ⚠️ Motion File Format Conversion [HIGH PRIORITY]
**Status**: DIFFERENT formats

**MJX**: LZ4-compressed motion files
**mjlab**: NPZ files with specific structure

**Conversion Script Needed**:
```python
# Pseudo-code for converter
def convert_mjx_motion_to_mjlab(mjx_motion_file, output_npz):
    # 1. Load LZ4 compressed data
    # 2. Extract: joint_pos, joint_vel, body_pos_w, body_quat_w, 
    #             body_lin_vel_w, body_ang_vel_w
    # 3. Save as NPZ with correct structure
    np.savez(output_npz, 
             joint_pos=..., 
             joint_vel=...,
             body_pos_w=...,
             body_quat_w=...,
             body_lin_vel_w=...,
             body_ang_vel_w=...)
```

**Note**: mjlab provides `csv_to_npz.py` for CSV input. You may need to:
1. Export MJX motions to CSV first
2. Use mjlab's converter, or
3. Write custom LZ4 → NPZ converter

**Estimated Effort**: 1 day for converter + testing

---

## Important Migration Tasks (Should Complete)

### 4. Action Delay Mechanism
**Status**: MISSING in mjlab

**MJX**: `n_steps_delay` parameter
**mjlab**: No built-in support

**Implementation Options**:
- Add delay buffer to `ActionManager`
- Store last N actions and apply with delay
- Add `delay_steps` parameter to `ActionTermCfg`

**Estimated Effort**: 1 day

---

### 5. Advanced IMU Noise Models
**Status**: PARTIAL support

**MJX Features**:
- AR(1) processes for gyro
- Bias random walks
- Amplitude variation
- Separate quaternion noise model

**mjlab Current**:
- Additive bias noise
- Gaussian/Uniform noise
- Simple noise model

**Migration Path**:
- Extend `NoiseModel` class with AR(1) implementation
- Add bias random walk support
- May not be critical if simpler noise is sufficient

**Estimated Effort**: 2-3 days (if needed)

---

### 6. DeepMimic Reward Functions
**Status**: MISSING specialized implementations

**Required Rewards**:
- `body_quat_tracking`: Track body quaternion orientations
- `site_pos_tracking`: Track end-effector positions
- Exponential tracking rewards with sigma

**Implementation**:
```python
# Add to src/mjlab/envs/mdp/rewards.py
def body_quat_tracking(env, target_quat, sigma=1.0):
    # Compute quaternion error
    # Return exp(-error^2 / sigma^2)
    pass

def site_pos_tracking(env, target_pos, sigma=1.0):
    # Track site positions
    pass
```

**Estimated Effort**: 2 days

---

### 7. Motion Phase Signal
**Status**: Only in tracking task, not general

**MJX**: Cyclic phase encoding for all periodic motions
**mjlab**: Limited to tracking task

**Implementation**: 
- Add phase computation to `ObservationManager`
- Make available as general observation term

**Estimated Effort**: 1 day

---

### 8. Specialized Environments
**Status**: MISSING (MJX has 27+, mjlab has 2 base tasks)

**MJX Environments Not in mjlab**:
- Crawl (multi-directional)
- Climb Wall
- Get Up (from prone)
- Stair Climb/Descend
- Box Climbing (8 variants)
- Chair environments
- Rotate Box
- Cartwheel

**Migration Decision Required**:
- Do you need these specific environments?
- Can they be generalized into velocity + tracking tasks?
- If needed, each environment is ~1-2 days to port

**Estimated Effort**: Variable (2-15 days depending on needs)

---

## Optional Migration Tasks (Nice to Have)

### 9. Recurrent Policy Networks (LSTM/GRU)
**Status**: NOT SUPPORTED

**Impact**: Low priority unless you specifically need recurrence
**Workaround**: Use frame stacking instead of LSTM memory

**Estimated Effort**: 3-4 days (requires RSL-RL extension)

---

### 10. Symmetry Augmentation / Mirror Loss
**Status**: NOT SUPPORTED

**Impact**: Can improve sample efficiency for bipedal locomotion
**Workaround**: May not be critical with enough training data

**Estimated Effort**: 2-3 days

---

### 11. ZMP Walking Reference Generator
**Status**: NOT SUPPORTED

**Impact**: Only needed if you want to generate walking gaits procedurally
**Workaround**: Use captured motion references instead

**Estimated Effort**: 3-5 days

---

## Configuration Migration Example

### Before (MJX + Gin):
```python
# config.gin
MJXConfig.episode_length = 1000
MJXConfig.action_scale = 0.25
MJXConfig.frame_stack = 15
RewardScales.lin_vel_tracking = 1.5
RewardScales.ang_vel_tracking = 0.8
```

### After (mjlab + Dataclass):
```python
# rough_env_cfg.py
@dataclass
class ToddlerBotRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Episode length
        self.episode_length_s = 20.0  # 1000 steps * 0.02s
        
        # Action scaling
        self.actions.joint_pos.scale = 0.25
        
        # Frame stacking - NOT YET SUPPORTED
        # self.observations.policy.frame_stack = 15
        
        # Reward weights
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.8
```

---

## Testing Strategy

### 1. Unit Testing
- Test individual components (rewards, observations, actions)
- Verify numerical equivalence where possible

### 2. Integration Testing
- Run short training episodes (100-1000 steps)
- Compare learning curves with MJX baseline
- Check that reward magnitudes are similar

### 3. Full Training Comparison
- Train for full duration on same task
- Compare final performance
- Measure sim-to-real transfer if applicable

---

## Migration Priority Recommendation

### Week 1: Critical Infrastructure
1. Implement frame stacking
2. Set up basic config conversion
3. Convert motion file formats
4. Test basic velocity task

### Week 2: Core Features
5. Port key reward functions
6. Add action delay if needed
7. Verify domain randomization equivalence
8. Test training pipeline end-to-end

### Week 3: Polish & Validation
9. Port specialized environments (if needed)
10. Add advanced noise models (if needed)
11. Full training validation
12. Performance tuning

---

## Known Limitations

### Cannot Port from MJX:
1. **JAX/JIT compilation benefits** - mjlab uses PyTorch
   - *Impact*: Potentially slower compile times, but MuJoCo Warp is highly optimized
   
2. **Exact numerical equivalence** - Different physics backends
   - *Impact*: Results will differ slightly but should be qualitatively similar

### mjlab Advantages Over MJX:
1. **Cleaner, more modular API** - Easier to extend
2. **Better debugging** - PyTorch ecosystem
3. **More robots supported** - G1, Go1, ToddlerBot out of box
4. **Better documentation** - Clearer codebase structure

---

## Questions to Answer Before Starting

1. **Which MJX environments do you actually use?**
   - Focus migration effort on used environments only

2. **Are advanced noise models critical for your sim-to-real transfer?**
   - If yes, prioritize AR(1) and bias random walk implementation

3. **Do you need recurrent policies?**
   - If yes, will need to extend RSL-RL integration

4. **What is your performance baseline?**
   - Establish MJX metrics before migration for comparison

5. **Do you need exact reproducibility or just equivalent performance?**
   - Sets expectations for validation criteria

---

## Getting Help

- **mjlab Issues**: https://github.com/mujocolab/mjlab/issues
- **Documentation**: See `docs/` folder in mjlab repo
- **Examples**: Check `src/mjlab/tasks/velocity` and `src/mjlab/tasks/tracking`

---

## Conclusion

The migration from MJX to mjlab is **feasible** with the main work being:
1. Frame stacking implementation (~1-2 days)
2. Config system conversion (~2-3 days per environment)
3. Motion file format conversion (~1 day)

**Total Estimated Effort**: 1-2 weeks for basic migration + 1-2 weeks for validation and tuning

The core RL infrastructure is solid in mjlab, so once the basics are ported, you should be able to match or exceed MJX performance with a cleaner, more maintainable codebase.




