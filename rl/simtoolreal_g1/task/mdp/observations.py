from __future__ import annotations

import math

import torch

from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
)

NUM_JOINTS: int = 13
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

PALM_CENTER_OFFSET: tuple[float, float, float] = (-0.0, -0.02, 0.16)
FINGERTIP_OFFSET: tuple[float, float, float] = (0.02, 0.002, 0.0)

KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}

def compute_obs_dim(field_list) -> int:
    return sum(OBS_FIELD_SIZES[f] for f in field_list)

def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )

def _full_body_state_w(data) -> torch.Tensor:
    return torch.cat(
        [
            data.body_link_pos_w,
            data.body_link_quat_w,
            data.body_link_lin_vel_w,
            data.body_link_ang_vel_w,
        ],
        dim=-1,
    )

def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(-max_deg, max_deg) * (
        math.pi / 180.0
    )
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)

def _apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset: tuple[float, float, float],
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    offset_t = offset_t.expand(*batch_shape, 3)
    shifted = quat_apply(
        rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)
    ).reshape(*batch_shape, 3)
    return pos_w + shifted

def _keypoints_world(
    center_pos: torch.Tensor,
    center_rot: torch.Tensor,
    kp_offsets: torch.Tensor,
) -> torch.Tensor:
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)

def _episode_start(env) -> torch.Tensor:
    return (env.episode_length_buf == 0) & (env._successes == 0)

def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    env,
    flush: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)

    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    delayed = queue[torch.arange(env.num_envs, device=env.device), idx]
    return queue, delayed

def _canonical_joint_obs(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = env._perm_lab_to_canon
    joint_pos_raw = env.scene["robot"].data.joint_pos[:, perm]
    joint_pos = (
        2.0 * (joint_pos_raw - env._joint_lower_canon)
        / (env._joint_upper_canon - env._joint_lower_canon)
        - 1.0
    )
    joint_vel = env.scene["robot"].data.joint_vel[:, perm]
    # _prev_targets is already 13-wide in canonical [arm, hand] order.
    return joint_pos, joint_vel, env._prev_targets

def compute_intermediate_values(env) -> None:
    """Update shared geometric state for rewards and terminations."""
    from .rewards import update_near_goal_steps  # local import to avoid cycle

    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins

    obj_pos = env.scene["object"].data.root_link_pos_w - env_origins
    obj_rot = env.scene["object"].data.root_link_quat_w
    goal_pos = env._goal_pos
    goal_rot = env._goal_rot

    body_state = _full_body_state_w(env.scene["robot"].data)
    ft_state = body_state[:, env._fingertip_body_ids, :]
    pad_off = env._fingertip_pad_offsets.unsqueeze(0).expand(env.num_envs, NUM_FINGERTIPS, 3)
    ft_pad_w = ft_state[:, :, 0:3] + quat_apply(ft_state[:, :, 3:7], pad_off)
    ft_pos = ft_pad_w - env_origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(
        ft_pos - obj_pos.unsqueeze(1), dim=-1
    )

    env._palm_pos = body_state[:, env._palm_body_id, 0:3] - env_origins

    c = env.sim.data.contact
    cg = c.geom.to(torch.long)
    cd = c.dist
    cw = c.worldid.to(torch.long)
    gbid = env._geom_bodyid
    valid = (cg[:, 0] >= 0) & (cg[:, 1] >= 0) & (cd < 0.005)
    b0 = gbid[cg[:, 0].clamp(min=0)]
    b1 = gbid[cg[:, 1].clamp(min=0)]
    obj0 = b0 == env._object_body_id
    obj1 = b1 == env._object_body_id
    ft_body = torch.where(obj0, b1, b0)  # the non-object side of the pair
    is_ft = torch.isin(ft_body, env._fingertip_body_ids_t)
    sel = valid & (obj0 | obj1) & is_ft
    contact_mat = torch.zeros(
        env.num_envs, NUM_FINGERTIPS, device=env.device, dtype=torch.bool
    )
    if bool(sel.any()):
        fidx = (
            ft_body[sel].unsqueeze(-1) == env._fingertip_body_ids_t
        ).float().argmax(-1)
        contact_mat[cw[sel], fidx] = True
    env._fingertip_contact = contact_mat
    env._n_fingers_contact = contact_mat.sum(-1).float()  # (N,) 0..5

    if rew_cfg.fixed_size_keypoint_reward:
        kp_offsets = env._keypoint_offsets_fixed
    else:
        kp_offsets = env._keypoint_offsets

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    env._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values

    sentinel = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(
        sentinel, env._keypoints_max_dist, env._closest_keypoint_max_dist
    )
    sentinel_ft = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(
        sentinel_ft, env._curr_fingertip_distances, env._closest_fingertip_dist
    )

    if hasattr(env, "_keypoint_success_tolerance_m"):
        tol = env._keypoint_success_tolerance_m()
    else:
        tol = env._current_success_tolerance * rew_cfg.keypoint_scale
    env._near_goal = env._keypoints_max_dist <= tol
    env._near_goal_steps = update_near_goal_steps(
        near_goal=env._near_goal,
        near_goal_steps=env._near_goal_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )
    env._is_success = env._near_goal_steps >= term_cfg.success_steps

def _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    dr = env.cfg.domain_randomization
    state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
    env._object_state_queue, delayed = _sample_delay(
        env._object_state_queue, state, env, flush=_episode_start(env)
    )
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]
    return noisy_pos, noisy_rot, noisy_vel

def _apply_obs_delay(env, policy_tensor: torch.Tensor) -> torch.Tensor:
    env._obs_queue, delayed = _sample_delay(
        env._obs_queue, policy_tensor, env, flush=_episode_start(env)
    )
    return delayed

def build_observations(env) -> dict[str, torch.Tensor]:

    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)

    body_state = _full_body_state_w(env.scene["robot"].data)

    palm_state = body_state[:, env._palm_body_id, :]  # (N, 13)
    palm_pos_w = palm_state[:, 0:3]
    palm_rot = palm_state[:, 3:7]  # wxyz
    palm_vel = palm_state[:, 7:13]

    palm_center_pos_w = _apply_local_offset(
        palm_pos_w, palm_rot, PALM_CENTER_OFFSET, (env.num_envs,)
    )
    palm_pos = palm_center_pos_w - env_origins

    ft_state = body_state[:, env._fingertip_body_ids, :]  # (N, 5, 13)
    ft_body_pos_w = ft_state[:, :, 0:3]
    ft_body_rot_w = ft_state[:, :, 3:7]

    ft_pos_w = _apply_local_offset(
        ft_body_pos_w,
        ft_body_rot_w,
        FINGERTIP_OFFSET,
        (env.num_envs, NUM_FINGERTIPS),
    )

    obj_pos = env.scene["object"].data.root_link_pos_w - env_origins
    obj_rot = env.scene["object"].data.root_link_quat_w
    obj_linvel = env.scene["object"].data.root_link_lin_vel_w
    obj_angvel = env.scene["object"].data.root_link_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)

    goal_pos = env._goal_pos
    goal_rot = env._goal_rot

    if dr.use_object_state_delay_noise:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = _apply_object_state_dr(
            env, obj_pos, obj_rot, obj_linvel, obj_angvel
        )
    else:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = obj_pos, obj_rot, obj_vel

    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)
    noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)

    goal_yaw_obs_noise = getattr(env, "goal_yaw_obs_noise", None)
    if goal_yaw_obs_noise is not None and torch.any(goal_yaw_obs_noise != 0):
        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], device=goal_rot.device, dtype=goal_rot.dtype
        ).unsqueeze(0).expand(env.num_envs, -1)
        yaw_q = quat_from_angle_axis(goal_yaw_obs_noise, z_axis)
        noisy_goal_rot = quat_mul(yaw_q, goal_rot)
        noisy_goal_kp = _keypoints_world(goal_pos, noisy_goal_rot, kp_offsets)
    else:
        noisy_goal_kp = goal_kp

    keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = noisy_obj_kp - noisy_goal_kp

    fingertip_pos_rel_palm = (
        (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1)
    )

    object_scales_obs = env._object_scale_per_env * env._object_scale_multiplier

    palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
    obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
    noisy_obj_rot_xyzw = convert_quat(noisy_obj_rot, to="xyzw")

    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot_xyzw,
        "palm_vel": palm_vel,
        "object_rot": obj_rot_xyzw,
        "object_vel": obj_vel,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm": keypoints_rel_palm_clean,
        "keypoints_rel_goal": keypoints_rel_goal_clean,
        "object_scales": object_scales_obs,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = (
            joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        )

    state_tensor = _stack_obs_dict(obs_clean, env.cfg.obs.state_list)
    policy_tensor = _stack_obs_dict(obs_noisy, env.cfg.obs.obs_list)

    if dr.use_obs_delay:
        policy_tensor = _apply_obs_delay(env, policy_tensor)

    clip = env.cfg.obs.clamp_abs_observations
    policy_tensor = policy_tensor.clamp(-clip, clip)
    state_tensor = state_tensor.clamp(-clip, clip)

    return {"policy": policy_tensor, "critic": state_tensor}


__all__ = [
    "NUM_JOINTS",
    "NUM_FINGERTIPS",
    "NUM_KEYPOINTS",
    "KEYPOINT_CORNERS",
    "OBS_FIELD_SIZES",
    "compute_obs_dim",
    "compute_intermediate_values",
    "build_observations",
]
