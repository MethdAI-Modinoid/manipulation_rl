"""Interactive MuJoCo viewer on the SimToolReal-G1 scene, posed exactly as the RL
env's ready stance (robot @ z=0.9, bent legs, right arm hovering over the object)
— plus a floor and light so it's actually visible. Fast: builds a standalone
model, no MuJoCo-Warp startup. Inspection only (no physics randomization, no
policy). This is the CLEAN reference pose; during training the arm starts jittered
by reset domain-randomization.

Run:  ./scripts/view.sh view_scene
"""
import re

import mujoco
import mujoco.viewer

from simtoolreal_g1.robot_cfg import INIT_STATE, get_spec as robot_spec
from simtoolreal_g1.task.env_cfg import EnvCfg


def build_model() -> mujoco.MjModel:
    cfg = EnvCfg()
    spec = robot_spec()  # pelvis already re-homed to z=0.9 in the XML
    wb = spec.worldbody

    wb.add_light(pos=[0.4, -0.3, 2.4], dir=[0, 0.1, -1],
                 diffuse=[0.9, 0.9, 0.9], specular=[0.3, 0.3, 0.3])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[3, 3, 0.1], pos=[0, 0, 0], rgba=[0.3, 0.3, 0.35, 1])

    tp, th = cfg.reset.table_pos, cfg.reset.table_half_extents
    tbl = wb.add_body(name="viz_table", pos=list(tp))
    tbl.add_geom(name="viz_table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=list(th), rgba=[0.55, 0.4, 0.25, 1])

    op = cfg.reset.object_start_pos
    obj = wb.add_body(name="viz_tool", pos=list(op))
    obj.add_freejoint()
    obj.add_geom(name="viz_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.05, 0.01, 0.0075], rgba=[0.6, 0.3, 0.1, 1], density=400)
    obj.add_geom(name="viz_head", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.015, 0.03, 0.015], pos=[0.065, 0, 0],
                 rgba=[0.5, 0.5, 0.5, 1], density=800)
    return spec.compile()


def apply_ready_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set each joint to its INIT_STATE ready-stance angle (regex-matched)."""
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        short = name.split("/")[-1]
        # Apply ALL matching patterns in order (last wins) — matches how mjlab's
        # Entity resolves joint_pos, so ".*":0.0 (first) is overridden by the
        # specific arm-joint angles (after). A first-match break would wrongly
        # leave the arm at the ".*" rest value.
        for pat, val in INIT_STATE.joint_pos.items():
            if re.fullmatch(pat, short):
                data.qpos[model.jnt_qposadr[jid]] = val


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    apply_ready_pose(model, data)
    mujoco.mj_forward(model, data)
    print(f"scene: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom}")
    print("Ready-pose reference view. Close the window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
