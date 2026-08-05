from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1c

REPO = Path(__file__).resolve().parents[2]
XML = REPO / "robot" / "g1_revo2_rl.xml"
assert XML.exists(), f"run build_rl_model.py first — missing {XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
    mesh_root = XML.parent / "meshes"
    assets: dict[str, bytes] = {}
    for f in mesh_root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(mesh_root).as_posix()
            assets[f"{meshdir}/{rel}"] = f.read_bytes()
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XML))
    # Normalize "meshes/" -> "meshes" so attach-prefixed asset keys match the
    # compiler's file lookup (a trailing slash yields double-slash key mismatch).
    spec.meshdir = spec.meshdir.rstrip("/")
    spec.assets = get_assets(spec.meshdir)
    return spec

# Revo2 hands: 6 ACTUATED finger joints per hand (thumb metacarpal + 5 proximal).
# The 5 distal joints per hand are passive (coupled), so not matched here.
HAND = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        r".*_proximal_joint",         # thumb/index/middle/ring/pinky proximal, both hands (10)
        r".*_thumb_metacarpal_joint",  # both hands (2)
    ),
    stiffness=20.0,
    damping=1.0,
    effort_limit=5.0,
    armature=0.01,  
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # Right arm (7): reuse Unitree G1's tuned groups. With the left arm + legs
        # welded, these regexes now match only the right-side joints.
        #   5020 -> elbow, shoulder pitch/roll/yaw, wrist_roll  (5)
        #   4010 -> wrist_pitch, wrist_yaw                       (2)
        g1c.G1_ACTUATOR_5020,
        g1c.G1_ACTUATOR_4010,
        # Right hand (6 actuated fingers).
        HAND,
    ),
    soft_joint_pos_limit_factor=0.9,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    # Fixed base: mjlab's Entity writes this onto the pelvis root body, so it must
    # match build_rl_model's re-home height (z=0.9).
    pos=(0.0, 0.0, 0.774),  
    joint_pos={
        "right_shoulder_pitch_joint": -0.5,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.8,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 1.2,
        "right_wrist_yaw_joint": 0.0,
        ".*": 0.0,  # fingers open + all welded joints at zero (fallback, matches last)
    },
    joint_vel={".*": 0.0},
)

def get_robot_cfg() -> EntityCfg:
    """Fresh EntityCfg for the fixed-base full-body G1 + Revo2 hands."""
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )

if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    robot = Entity(get_robot_cfg())
    model = robot.spec.compile()
    print(f"actuators: {model.nu}  (expected 13)")
    print(f"joints: {model.njnt}  DOF: {model.nv}")
    assert model.nu == 13, f"expected 13 actuators (arm 7 + hand 6), got {model.nu}"
    print("OK")
