"""ToddlerBot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

TODDLERBOT_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "toddlerbot_2xc"
  / "xmls"
  / "toddlerbot_2xc.xml"
)
assert TODDLERBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, TODDLERBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TODDLERBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.02020514, 0.0, 0.310053),
  joint_pos={
    "left_hip_pitch": -0.091312,
    "left_knee": -0.380812,
    "left_ankle_pitch": -0.2895,
    "right_hip_pitch": 0.091312,
    "right_knee": 0.380812,
    "right_ankle_pitch": 0.2895,
    "left_shoulder_pitch": 0.174533,
    ".*_shoulder_roll": 0.087266,
    "left_shoulder_yaw_drive": 1.570796,
    "left_shoulder_yaw_driven": -1.570796,
    ".*_elbow_roll": -0.523599,
    "left_elbow_yaw_drive": -1.570796,
    "left_elbow_yaw_driven": 1.570796,
    "left_wrist_pitch_drive": 1.22173,
    "left_wrist_pitch_driven": -1.22173,
    "right_shoulder_pitch": -0.174533,
    "right_shoulder_yaw_drive": -1.570796,
    "right_shoulder_yaw_driven": 1.570796,
    "right_elbow_yaw_drive": 1.570796,
    "right_elbow_yaw_driven": -1.570796,
    "right_wrist_pitch_drive": -1.22173,
    "right_wrist_pitch_driven": 1.22173,
  },
  joint_vel={".*": 0.0},
  ctrl={
    "left_hip_pitch": -0.091312,
    "left_knee": -0.380812,
    "left_ankle_pitch": -0.2895,
    "right_hip_pitch": 0.091312,
    "right_knee": 0.380812,
    "right_ankle_pitch": 0.2895,
    "left_shoulder_pitch": 0.174533,
    ".*_shoulder_roll": 0.087266,
    "left_shoulder_yaw_drive": 1.570796,
    ".*_elbow_roll": -0.523599,
    "left_elbow_yaw_drive": -1.570796,
    "left_wrist_pitch_drive": 1.22173,
    "right_shoulder_pitch": -0.174533,
    "right_shoulder_yaw_drive": -1.570796,
    "right_elbow_yaw_drive": 1.570796,
    "right_wrist_pitch_drive": -1.22173,
  },
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={r"^(left|right)_ankle_roll_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_ankle_roll_collision$": 1},
  friction={r"^(left|right)_ankle_roll_collision$": (0.6,)},  # TODO: tune friction
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_ankle_roll_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_ankle_roll_collision$": 1},
  friction={r"^(left|right)_ankle_roll_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[r"^(left|right)_ankle_roll_collision$"],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

TODDLERBOT_ROBOT_CFG = EntityCfg(
  init_state=HOME_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
)

TODDLERBOT_ACTION_SCALE = {".*": 0.25}

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(TODDLERBOT_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
