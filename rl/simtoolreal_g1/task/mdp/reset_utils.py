from __future__ import annotations

import math

import torch

from mjlab.third_party.isaaclab.isaaclab.utils.math import random_orientation

from .actions import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .observations import KEYPOINT_CORNERS, NUM_FINGERTIPS, compute_obs_dim
from ..objects.object_gen import has_head
from ..objects.object_size_distributions import distributions_for_topology

OBJECT_DR_FIELDS = (
    "geom_size", "geom_pos", "body_mass", "body_inertia", "body_ipos", "body_iquat",
)

# Canonical RL joint order = arm then hand.
ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
HAND_JOINT_NAMES = (
    "right_thumb_metacarpal_joint", "right_thumb_proximal_joint",
    "right_index_proximal_joint", "right_middle_proximal_joint",
    "right_ring_proximal_joint", "right_pinky_proximal_joint",
)
FINGERTIP_BODY_NAMES = (
    "right_thumb_distal_link", "right_index_distal_link",
    "right_middle_distal_link", "right_ring_distal_link", "right_pinky_distal_link",
)
# Tactile-PAD offsets in each distal-link frame (from the Revo2 URDF). Grasp is
# measured from the PADS, not the distal body origins (knuckles), else a splayed
# hovering hand fakes a grasp. Order matches FINGERTIP_BODY_NAMES.
FINGERTIP_PAD_OFFSETS = (
    (0.0,       0.0137928, 0.0076766),
    (0.014497,  0.0,       0.023303),
    (0.0159759, 0.0,       0.0265894),
    (0.0154392, 0.0,       0.0260561),
    (0.01401,   0.0,       0.021124),
)
PALM_BODY_NAME = "right_hand_mount"

# Steps to let a freshly-reset object fall, tumble and settle before capturing its
# rest height as the lifting reference (x decimation 2 = 50 physics steps @ 120 Hz
# ~= 0.42 s). The arm starts ~10 cm away so it won't touch the object this early.
SETTLE_STEPS = 25

def sample_object_sizes(env, topology: str) -> None:
    """Sample per-env tool dims/densities from the TOPOLOGY's distribution subset.

    Object identity is fixed per env for the run. Handle full-length convention:
    box -> (Lx, Ly, Lz); cylinder -> (L, D, D)."""
    n = env.num_envs
    dev = env.device

    # BOOTSTRAP: single fixed tall graspable bar for every env (no distribution).
    obj_cfg = env.cfg.object
    if getattr(obj_cfg, "bootstrap_grasp", False):
        full = torch.tensor(obj_cfg.bootstrap_handle_full, dtype=torch.float32)
        handle_len = full.unsqueeze(0).expand(n, 3).clone()
        head_len = torch.full((n, 3), 1e-4)                 # ~no head
        vol = handle_len[:, 0] * handle_len[:, 1] * handle_len[:, 2]
        handle_mass = vol * obj_cfg.bootstrap_density
        env._object_topology = topology
        env._object_handle_len = handle_len.to(dev)
        env._object_head_len = head_len.to(dev)
        env._object_handle_half = (0.5 * handle_len).to(dev)
        env._object_head_half = (0.5 * head_len).to(dev)
        env._object_handle_mass = handle_mass.to(dev)
        env._object_head_mass = torch.zeros(n).to(dev)
        env._object_mass = handle_mass.clamp_min(1e-4).unsqueeze(-1).to(dev)
        base = env.cfg.reward.object_base_size
        env._object_scale_per_env = (handle_len / base).to(dev)
        return

    dists = distributions_for_topology(topology)

    # Assign each env a distribution round-robin, then sample its dims.
    idx = torch.arange(n) % len(dists)
    handle_len = torch.zeros(n, 3)
    head_len = torch.zeros(n, 3)
    handle_density = torch.zeros(n)
    head_density = torch.zeros(n)
    for di, dist in enumerate(dists):
        sel = (idx == di).nonzero(as_tuple=False).squeeze(-1)
        if sel.numel() == 0:
            continue
        k = sel.numel()
        hs = torch.as_tensor(dist.sample_handle_scales(k), dtype=torch.float32)
        if hs.shape[1] == 2:  # cylinder (length, diameter) -> (L, D, D)
            hs = torch.stack([hs[:, 0], hs[:, 1], hs[:, 1]], dim=-1)
        handle_len[sel] = hs
        handle_density[sel] = torch.as_tensor(
            dist.sample_handle_densities(k), dtype=torch.float32
        )
        heads = dist.sample_head_scales(k)
        if heads is not None:
            head_len[sel] = torch.as_tensor(heads, dtype=torch.float32)
            head_density[sel] = torch.as_tensor(
                dist.sample_head_densities(k), dtype=torch.float32
            )
        else:
            head_len[sel] = 1e-4  # ~no head (eraser)
            head_density[sel] = 0.0

    is_cyl = topology == "cyl_head"
    if is_cyl:
        handle_vol = math.pi * (0.5 * handle_len[:, 1]) ** 2 * handle_len[:, 0]
    else:
        handle_vol = handle_len[:, 0] * handle_len[:, 1] * handle_len[:, 2]
    head_vol = head_len[:, 0] * head_len[:, 1] * head_len[:, 2]
    if not has_head(topology):
        head_vol = torch.zeros_like(head_vol)
    handle_mass = handle_vol * handle_density
    head_mass = head_vol * head_density

    env._object_topology = topology
    env._object_handle_len = handle_len.to(dev)  # full lengths
    env._object_head_len = head_len.to(dev)
    env._object_handle_half = (0.5 * handle_len).to(dev)
    env._object_head_half = (0.5 * head_len).to(dev)
    env._object_handle_mass = handle_mass.to(dev)
    env._object_head_mass = head_mass.to(dev)
    env._object_mass = (handle_mass + head_mass).clamp_min(1e-4).unsqueeze(-1).to(dev)
    # Normalized handle scale (full length / object_base_size) for obs + keypoints.
    base = env.cfg.reward.object_base_size
    env._object_scale_per_env = (handle_len / base).to(dev)

def _object_body_geom_ids(env):
    """(handle global geom id, head global geom id | None, object body id).

    Matches by spec geom name (suffix-robust against any attach prefix)."""
    obj = env.scene["object"]
    gid = {}
    for i, g in enumerate(obj.indexing.geoms):
        gid[g.name.split("/")[-1]] = int(obj.indexing.geom_ids[i])
    handle = gid["handle"]
    head = gid.get("head")
    return handle, head, int(obj.indexing.root_body_id)

def _handle_own_inertia(env, topology: str) -> torch.Tensor:
    """Diagonal inertia of the handle about its own center (body origin)."""
    hl = env._object_handle_len
    m = env._object_handle_mass
    if topology == "cyl_head":
        r = 0.5 * hl[:, 1]
        L = hl[:, 0]
        ix = 0.5 * m * r ** 2               # axial (x)
        iyz = m * (3.0 * r ** 2 + L ** 2) / 12.0
        return torch.stack([ix, iyz, iyz], dim=-1)
    lx, ly, lz = hl[:, 0], hl[:, 1], hl[:, 2]
    ix = m * (ly ** 2 + lz ** 2) / 12.0
    iy = m * (lx ** 2 + lz ** 2) / 12.0
    iz = m * (lx ** 2 + ly ** 2) / 12.0
    return torch.stack([ix, iy, iz], dim=-1)

def _head_own_inertia(env) -> torch.Tensor:
    hl = env._object_head_len
    m = env._object_head_mass
    hx, hy, hz = hl[:, 0], hl[:, 1], hl[:, 2]
    ix = m * (hy ** 2 + hz ** 2) / 12.0
    iy = m * (hx ** 2 + hz ** 2) / 12.0
    iz = m * (hx ** 2 + hy ** 2) / 12.0
    return torch.stack([ix, iy, iz], dim=-1)

def write_object_geometry(env, topology: str) -> None:

    dev = env.device
    n = env.num_envs
    w = torch.arange(n, device=dev)
    handle_gid, head_gid, body_gid = _object_body_geom_ids(env)

    gs = env.sim.model.geom_size  # (n, ngeom, 3)
    hl = env._object_handle_len
    if topology == "cyl_head":
        gs[w, handle_gid, 0] = 0.5 * hl[:, 1]  # radius
        gs[w, handle_gid, 1] = 0.5 * hl[:, 0]  # half-length
        gs[w, handle_gid, 2] = 0.0
    else:
        gs[w, handle_gid, :] = 0.5 * hl        # box half-extents

    m_h = env._object_handle_mass
    i_h = _handle_own_inertia(env, topology)
    if has_head(topology):
        head_half = env._object_head_half
        gs[w, head_gid, :] = head_half
        head_x = 0.5 * hl[:, 0] + head_half[:, 0]  # head at +x end of handle
        gp = env.sim.model.geom_pos
        gp[w, head_gid, 0] = head_x
        gp[w, head_gid, 1] = 0.0
        gp[w, head_gid, 2] = 0.0
        m_head = env._object_head_mass
        i_head = _head_own_inertia(env)
    else:
        head_x = torch.zeros(n, device=dev)
        m_head = torch.zeros(n, device=dev)
        i_head = torch.zeros(n, 3, device=dev)

    total_m = (m_h + m_head).clamp_min(1e-6)
    com_x = m_head * head_x / total_m           # handle center at x=0
    dh2 = com_x ** 2                            # handle offset^2 from COM
    dhead2 = (head_x - com_x) ** 2              # head offset^2 from COM
    ixx = i_h[:, 0] + i_head[:, 0]              # x-axis: no parallel-axis term
    iyy = i_h[:, 1] + m_h * dh2 + i_head[:, 1] + m_head * dhead2
    izz = i_h[:, 2] + m_h * dh2 + i_head[:, 2] + m_head * dhead2
    # Floor principal inertia for stable integration (see ObjectCfg.object_min_inertia).
    ifloor = env.cfg.object.object_min_inertia
    ixx = ixx.clamp_min(ifloor)
    iyy = iyy.clamp_min(ifloor)
    izz = izz.clamp_min(ifloor)

    bm = env.sim.model.body_mass
    bi = env.sim.model.body_inertia
    bp = env.sim.model.body_ipos
    bq = env.sim.model.body_iquat
    bm[w, body_gid] = total_m
    bi[w, body_gid, 0] = ixx
    bi[w, body_gid, 1] = iyy
    bi[w, body_gid, 2] = izz
    bp[w, body_gid, 0] = com_x
    bp[w, body_gid, 1] = 0.0
    bp[w, body_gid, 2] = 0.0
    bq[w, body_gid, 0] = 1.0
    bq[w, body_gid, 1] = 0.0
    bq[w, body_gid, 2] = 0.0
    bq[w, body_gid, 3] = 0.0

    env.sim.forward()

def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache the hooks read. Call once in
    __init__ after the scene/sim exist and sample_object_sizes() has run."""
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward
    robot = env.scene["robot"]
    dev = env.device

    # --- Index caches ---
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    hand_ids = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)[0]
    env._arm_joint_ids = torch.tensor(arm_ids, device=dev, dtype=torch.long)   # 51-space (7)
    env._hand_joint_ids = torch.tensor(hand_ids, device=dev, dtype=torch.long)  # 51-space (6)
    env._rl_joint_ids = torch.cat([env._arm_joint_ids, env._hand_joint_ids])    # 51-space (13)
    env._palm_body_id = robot.find_bodies([PALM_BODY_NAME])[0][0]
    env._fingertip_body_ids = robot.find_bodies(list(FINGERTIP_BODY_NAMES), preserve_order=True)[0]
    assert len(env._fingertip_body_ids) == NUM_FINGERTIPS
    # observations rotates these into world for the true fingertip-PAD positions.
    env._fingertip_pad_offsets = torch.tensor(
        FINGERTIP_PAD_OFFSETS, device=dev, dtype=torch.float32
    )  # (5, 3)

    # Contact-based grasp caches: geom id -> body id, plus object/fingertip body ids,
    # so observations can count distinct fingertips TOUCHING the object each step.
    _gbid = env.sim.model.geom_bodyid
    _gbid = _gbid if torch.is_tensor(_gbid) else torch.as_tensor(_gbid)
    env._geom_bodyid = _gbid.to(dev).to(torch.long).flatten()
    env._fingertip_body_ids_t = torch.as_tensor(
        [int(x) for x in env._fingertip_body_ids], device=dev, dtype=torch.long
    )
    _handle_gid, _, _ = _object_body_geom_ids(env)
    env._object_body_id = int(env._geom_bodyid[_handle_gid])

    # Obs joint selection == RL joints in canonical order (13, into 51).
    env._perm_lab_to_canon = env._rl_joint_ids
    env._perm_canon_to_lab = torch.arange(13, device=dev)  # policy order == canonical

    limits = robot.data.joint_pos_limits  # (N, 51, 2)
    env._joint_lower_canon = limits[0, env._rl_joint_ids, 0]  # (13,)
    env._joint_upper_canon = limits[0, env._rl_joint_ids, 1]
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- RL action-target buffers (13-wide, canonical [arm, hand]) ---
    default = robot.data.default_joint_pos  # (N, 51)
    rl_default = default[:, env._rl_joint_ids].clone()  # (N, 13)
    env._cur_targets = rl_default.clone()
    env._prev_targets = rl_default.clone()

    # --- Keypoint offsets (object local frame) ---
    corners = torch.tensor(KEYPOINT_CORNERS, device=dev, dtype=torch.float32)  # (4, 3)
    env._keypoint_offsets = corners.unsqueeze(0) * (
        env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5
    ).unsqueeze(1)  # (N, 4, 3)
    env._keypoint_offsets_fixed = (
        corners * (0.5 * rew.keypoint_scale * torch.tensor(rew.fixed_size, device=dev))
    ).unsqueeze(0).expand(env.num_envs, -1, -1).contiguous()

    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(env.num_envs, 3, device=dev).uniform_(lo, hi)

    # --- Reward / termination trackers ---
    env._lifted_object = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    env._closest_keypoint_max_dist = torch.full((env.num_envs,), -1.0, device=dev)
    env._closest_fingertip_dist = torch.full((env.num_envs, NUM_FINGERTIPS), -1.0, device=dev)
    # Closure ratchet: max #fingertips ever in contact this episode.
    env._max_fingers_contact = torch.zeros(env.num_envs, device=dev)
    env._successes = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    env._near_goal_steps = torch.zeros(env.num_envs, dtype=torch.long, device=dev)

    # --- Tolerance curriculum ---
    env._current_success_tolerance = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    env._frame_counter = 0
    env._last_curriculum_update = 0

    # --- Workspace-box bounds (env-local, on the palm) ---
    rc = env.cfg.reward
    env._workspace_box_min = torch.tensor(rc.workspace_box_min, device=dev)
    env._workspace_box_max = torch.tensor(rc.workspace_box_max, device=dev)

    # --- Object lifted-reward reference z + goal buffers ---
    # object_init_z is captured after the object settles (see update_object_init_z);
    # seeded provisionally with the spawn z. _settle_counter counts down per reset.
    env._object_init_z = torch.full((env.num_envs,), float(env.cfg.reset.object_start_pos[2]), device=dev)
    # Per-env cube spawn (env-local xyz), so the lift goal tracks the scattered cube.
    env._object_spawn_local = torch.tensor(
        env.cfg.reset.object_start_pos, device=dev
    ).unsqueeze(0).expand(env.num_envs, -1).clone()
    env._settle_counter = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    env._goal_pos = torch.zeros(env.num_envs, 3, device=dev)
    env._goal_rot = torch.zeros(env.num_envs, 4, device=dev)
    env._goal_rot[:, 0] = 1.0

    # --- DR rolling buffers ---
    obs_dim = compute_obs_dim(env.cfg.obs.obs_list)
    env._object_state_queue = torch.zeros(env.num_envs, max(1, dr.object_state_delay_max), 13, device=dev)
    env._obs_queue = torch.zeros(env.num_envs, max(1, dr.obs_delay_max), obs_dim, device=dev)
    env._action_queue = torch.zeros(env.num_envs, max(1, dr.action_delay_max), 13, device=dev)

    # --- Wrench DR ---
    env._random_force_prob = sample_log_uniform(dr.force_prob_range, env.num_envs).to(dev)
    env._random_torque_prob = sample_log_uniform(dr.torque_prob_range, env.num_envs).to(dev)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=dev)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=dev)
    # env._object_mass already set by sample_object_sizes.

    # --- Step-shared caches (filled by compute_intermediate_values) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=dev)
    env._curr_fingertip_distances = torch.zeros(env.num_envs, NUM_FINGERTIPS, device=dev)
    env._near_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    env._is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)

    env.reward_buf = torch.zeros(env.num_envs, device=dev)

def _randomize_robot_dof_state(env, env_ids: torch.Tensor, alpha: float = 0.0) -> None:
    """Reset DOF state to the ANNEALED start pose and seed targets.

    Arm start = lerp(grasp_arm_pose, home_arm_pose, alpha): alpha~0 starts in the b1
    top-down grasp pose over the cradled cube; alpha~1 starts in the neutral away pose
    so the policy must REACH+reorient first. Fingers always start OPEN (lower limit).
    Start-pose noise grows with alpha."""
    cfg = env.cfg.reset
    robot = env.scene["robot"]
    default_pos = robot.data.default_joint_pos[env_ids].clone()  # (n, 51)
    lower = robot.data.joint_pos_limits[env_ids, :, 0]
    upper = robot.data.joint_pos_limits[env_ids, :, 1]
    arm, hand = env._arm_joint_ids, env._hand_joint_ids

    joint_pos = default_pos.clone()
    # Order matches ARM_JOINT_NAMES.
    grasp = torch.tensor(cfg.grasp_arm_pose, device=env.device, dtype=joint_pos.dtype)
    home = torch.tensor(cfg.home_arm_pose, device=env.device, dtype=joint_pos.dtype)
    start = torch.lerp(grasp, home, float(alpha))  # (7,)
    joint_pos[:, arm] = start.unsqueeze(0)
    joint_pos[:, hand] = lower[:, hand]  # fingers fully OPEN

    an = cfg.reset_dof_pos_random_interval_arm * float(alpha)
    fn = cfg.reset_dof_pos_random_interval_fingers * float(alpha)
    if an > 0.0:
        joint_pos[:, arm] += torch.empty(env_ids.numel(), arm.numel(), device=env.device).uniform_(-an, an)
    if fn > 0.0:
        # finger noise only toward closing (positive) so they never start past the open limit
        joint_pos[:, hand] += torch.empty(env_ids.numel(), hand.numel(), device=env.device).uniform_(0.0, fn)
    joint_pos = joint_pos.clamp(lower, upper)

    joint_vel = torch.zeros_like(default_pos)
    if cfg.reset_dof_vel_random_interval > 0.0:
        joint_vel[:, env._rl_joint_ids] = torch.empty_like(
            joint_vel[:, env._rl_joint_ids]
        ).uniform_(-cfg.reset_dof_vel_random_interval, cfg.reset_dof_vel_random_interval)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Seed RL targets from the reset pose so the servo holds it until the policy's
    # first action; held joints keep their default.
    env._prev_targets[env_ids] = joint_pos[:, env._rl_joint_ids]
    env._cur_targets[env_ids] = joint_pos[:, env._rl_joint_ids]
    full_target = default_pos.clone()
    full_target[:, env._rl_joint_ids] = joint_pos[:, env._rl_joint_ids]
    robot.set_joint_position_target(full_target, joint_ids=slice(None), env_ids=env_ids)

def _reset_object_pose(env, env_ids: torch.Tensor, alpha: float = 0.0) -> None:

    cfg = env.cfg.reset
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]
    base = torch.tensor(cfg.object_start_pos, device=env.device)
    noise = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0) * float(alpha)
    pos_local = base.unsqueeze(0) + noise * torch.tensor(
        [cfg.reset_position_noise_x, cfg.reset_position_noise_y, cfg.reset_position_noise_z],
        device=env.device,
    )
    if cfg.object_upright_spawn:
        # Upright with an annealed yaw about z, so the policy must rotate the wrist
        # to align its grasp.
        yaw_range = getattr(cfg, "object_yaw_range", 0.0) * float(alpha)
        yaw = (torch.empty(n, device=env.device).uniform_(-yaw_range, yaw_range)
               if yaw_range > 0.0 else torch.zeros(n, device=env.device))
        quat = torch.zeros(n, 4, device=env.device)
        quat[:, 0] = torch.cos(0.5 * yaw)  # wxyz: rotation about +z
        quat[:, 3] = torch.sin(0.5 * yaw)
    else:
        quat = random_orientation(n, device=str(env.device))

    env._object_init_z[env_ids] = pos_local[:, 2]  # provisional; settled below
    env._object_spawn_local[env_ids] = pos_local   # goal tracks the (scattered) cube
    env._settle_counter[env_ids] = SETTLE_STEPS

    pose = torch.cat([pos_local + origins, quat], dim=-1)
    obj = env.scene["object"]
    obj.data.write_root_pose(pose, env_ids=env_ids)
    obj.data.write_root_velocity(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

def update_object_init_z(env) -> None:
    pending = env._settle_counter > 0
    if not bool(pending.any()):
        return
    env._settle_counter[pending] -= 1
    just_settled = pending & (env._settle_counter == 0)
    if bool(just_settled.any()):
        obj_z = (
            env.scene["object"].data.root_link_pos_w - env.scene.env_origins
        )[:, 2]
        env._object_init_z[just_settled] = obj_z[just_settled]

def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    cfg = env.cfg.reset
    n = env_ids.numel()
    if cfg.easy_goal:
        gp = env._object_spawn_local[env_ids].clone()
        gp[:, 2] = gp[:, 2] + cfg.easy_goal_lift
        env._goal_pos[env_ids] = gp
        q = torch.zeros(n, 4, device=env.device)
        q[:, 0] = 1.0
        env._goal_rot[env_ids] = q
        return
    if mode == "delta":
        new_pos, new_quat = sample_delta_goal_pose(
            prev_pos=env._goal_pos[env_ids], prev_quat_wxyz=env._goal_rot[env_ids],
            delta_distance=cfg.delta_goal_distance, delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins, maxs=cfg.target_volume_maxs, scale=cfg.target_volume_region_scale,
        )
    elif mode == "absolute":
        new_pos, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins, maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale, n_envs=n, device=env.device,
        )
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")
    env._goal_pos[env_ids] = new_pos
    env._goal_rot[env_ids] = new_quat

def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0
    if hasattr(env, "_max_fingers_contact"):
        env._max_fingers_contact[env_ids] = 0.0

def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    _clear_goal_trackers(env, env_ids)
    # getattr default keeps old pickled cfg snapshots (which lack this field) playable.
    mode = getattr(env.cfg.reset, "goal_sampling_type", "absolute")
    _reset_goal_pose(env, env_ids, mode=mode)

def reset_env_state(env, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    rc = env.cfg.reset
    if getattr(rc, "eval_pose_alpha", None) is not None:
        alpha = float(rc.eval_pose_alpha)  # play/eval override (fresh env -> full task)
    else:
        anneal = getattr(rc, "pose_anneal_frames", 0)
        alpha = min(1.0, env._frame_counter / float(anneal)) if anneal and anneal > 0 else 1.0
    _randomize_robot_dof_state(env, env_ids, alpha)
    _reset_object_pose(env, env_ids, alpha)
    _reset_goal_pose(env, env_ids, mode="absolute")

    env._prev_episode_successes[env_ids] = env._successes[env_ids]
    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = False
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(dr.force_prob_range, n).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(dr.torque_prob_range, n).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(n, 3, device=env.device).uniform_(lo, hi)

__all__ = [
    "ARM_JOINT_NAMES", "HAND_JOINT_NAMES", "FINGERTIP_BODY_NAMES", "PALM_BODY_NAME",
    "OBJECT_DR_FIELDS", "sample_object_sizes", "write_object_geometry",
    "allocate_state_buffers", "reset_env_state", "reset_goal_trackers",
]
