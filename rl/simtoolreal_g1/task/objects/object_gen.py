"""Procedural tool object, parametrized by TOPOLOGY.

The 6 SimToolReal tool types reduce to 3 distinct MuJoCo-Warp topologies (Warp
shares geom_type across worlds, so genuinely different handle shapes need
different compiled models):

  - "box_head":  BOX handle      + BOX head   (hammer/screwdriver/spatula/brush, cuboid)
  - "cyl_head":  CYLINDER handle + BOX head   (marker + cylinder variants)
  - "box_only":  BOX handle,      no head     (eraser)

Within a topology, per-env size/mass variety is written into geom_size /
geom_pos / body_mass at startup (see reset_utils.write_object_geometry). The
spec sizes below are only placeholders that those writes overwrite.

Convention: the handle long axis is +x (head sits at the +x end), matching the
keypoint/reward code. The cylinder's local +z is rotated onto world +x via a
fixed quaternion so it, too, lies along x; only its radius/half-length vary
per world.
"""
from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

TOPOLOGIES = ("box_head", "cyl_head", "box_only")

# Placeholder half-extents (overwritten per world at startup).
_HANDLE_HALF = (0.05, 0.01, 0.0075)   # box handle (hx, hy, hz)
_HANDLE_CYL = (0.0075, 0.05)          # cylinder handle (radius, half_len)
_HEAD_HALF = (0.015, 0.03, 0.015)
_HEAD_POS = (0.065, 0.0, 0.0)
# Rotate cylinder local +z onto world +x: quat = (cos45, 0, sin45, 0).
_CYL_QUAT_Z_TO_X = (0.70710678, 0.0, 0.70710678, 0.0)


def box_mass(half_extents, density: float) -> float:
    hx, hy, hz = half_extents
    return (2 * hx) * (2 * hy) * (2 * hz) * density


def has_head(topology: str) -> bool:
    return topology in ("box_head", "cyl_head")


def get_bootstrap_cube_spec(half=0.03, mass=0.04) -> mujoco.MjSpec:
    """Dead-simple fixed cube (matches the labmate's working object): the geom has an
    EXPLICIT mass so the compiler computes a correct uniform-box inertia. No runtime
    resize, no analytic inertia, no inertia floor — that machinery is what made the
    table contact unstable for a light object. condim 6 + high friction like theirs."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tool")
    free = body.add_freejoint()
    free.name = "object_free_joint"
    g = body.add_geom(
        name="handle",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, half],
        mass=mass,
        rgba=[0.55, 0.27, 0.07, 1.0],
        friction=[3.0, 0.05, 0.01],
    )
    g.condim = 6
    return spec


def get_object_spec(topology: str = "box_head") -> mujoco.MjSpec:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}, expected one of {TOPOLOGIES}")
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tool")
    free = body.add_freejoint()
    free.name = "object_free_joint"

    # High sliding friction holds the object. condim=3 (not 6): halves constraint
    # rows per contact -> headroom for the raised buffer (see njmax/nconmax).
    _GRIP_FRICTION = [3.0, 0.05, 0.01]
    if topology == "cyl_head":
        g = body.add_geom(
            name="handle",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[_HANDLE_CYL[0], _HANDLE_CYL[1], 0.0],
            quat=list(_CYL_QUAT_Z_TO_X),
            rgba=[0.55, 0.27, 0.07, 1.0],
            density=400.0,
            friction=_GRIP_FRICTION,
        )
    else:  # box_head, box_only
        g = body.add_geom(
            name="handle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(_HANDLE_HALF),
            rgba=[0.55, 0.27, 0.07, 1.0],
            density=400.0,
            friction=_GRIP_FRICTION,
        )
    g.condim = 3
    # NOTE: default contact (no solref/solimp). The cube is light (~tens of g), so an
    # over-stiff contact (high solimp) makes the contact frequency too high for the
    # 120Hz step -> it goes UNSTABLE (bounces + penetrates). Defaults are stable.

    if has_head(topology):
        body.add_geom(
            name="head",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(_HEAD_HALF),
            pos=list(_HEAD_POS),
            rgba=[0.5, 0.5, 0.5, 1.0],
            density=800.0,
        )
    return spec


def get_object_cfg(topology: str = "box_head", start_pos=(0.15, -0.15, 1.0),
                   bootstrap_cube=None) -> EntityCfg:
    """bootstrap_cube = (half, mass) -> use the simple fixed cube (compiler inertia,
    no runtime resize). None -> the resizable multi-tool placeholder spec."""
    if bootstrap_cube is not None:
        half, mass = bootstrap_cube
        spec_fn = lambda: get_bootstrap_cube_spec(half, mass)
    else:
        spec_fn = lambda: get_object_spec(topology)
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(start_pos)),
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    for topo in TOPOLOGIES:
        m = Entity(get_object_cfg(topo)).spec.compile()
        handle_gid = m.geom("handle").id
        htype = m.geom_type[handle_gid]
        expect_cyl = topo == "cyl_head"
        is_cyl = htype == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert is_cyl == expect_cyl, f"{topo}: handle type mismatch"
        assert m.nv == 6, f"{topo}: expected 6 free dof, got {m.nv}"
        expect_geoms = 2 if has_head(topo) else 1
        assert m.ngeom == expect_geoms, f"{topo}: expected {expect_geoms} geoms, got {m.ngeom}"
        print(f"{topo:9s} geoms={m.ngeom} handle_type={'cyl' if is_cyl else 'box'} OK")
