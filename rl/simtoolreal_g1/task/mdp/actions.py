from __future__ import annotations
import math
import torch

def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )

def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Apply delay, canonical-to-Lab mapping, smoothing, and target clamps."""
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt

    actions = actions.clone().to(env.device)
    actions = actions[:, env._perm_canon_to_lab]

    if dr.use_action_delay and dr.action_delay_max > 0:
        episode_start = (env.episode_length_buf == 0) & (env._successes == 0)
        if episode_start.any():
            env._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    # Arm: velocity-delta accumulator (7 joints).
    arm_action = actions[:, :7]
    arm_raw = env._prev_targets[:, :7] + act_cfg.dof_speed_scale * dt * arm_action
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm_smoothed = (
        act_cfg.arm_moving_average * arm_raw
        + (1.0 - act_cfg.arm_moving_average) * env._prev_targets[:, :7]
    )
    arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)

    # Hand: absolute [-1, 1] scale (6 actuated fingers).
    hand_action = actions[:, 7:]
    hand_raw = env._hand_lower + 0.5 * (hand_action + 1.0) * (
        env._hand_upper - env._hand_lower
    )
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * env._prev_targets[:, 7:]
    )
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # _cur_targets is 13-wide, canonical order [arm 7, hand 6].
    env._cur_targets[:, :7] = arm_smoothed
    env._cur_targets[:, 7:] = hand_smoothed
    env._prev_targets = env._cur_targets.clone()


def apply_wrench_dr(env) -> None:
    """Apply decayed random force/torque impulses to the object."""
    dr = env.cfg.domain_randomization
    dt_pol = env.step_dt 

    if dr.force_decay > 0.0:
        env._object_forces *= dr.force_decay ** (dt_pol / dr.force_decay_interval)
    else:
        env._object_forces.zero_()
    if dr.torque_decay > 0.0:
        env._object_torques *= dr.torque_decay ** (dt_pol / dr.torque_decay_interval)
    else:
        env._object_torques.zero_()

    force_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_force_prob
    ).view(-1, 1, 1)
    torque_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_torque_prob
    ).view(-1, 1, 1)
    mass = env._object_mass.unsqueeze(-1)
    new_force = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.force_scale
    )
    new_torque = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.torque_scale
    )
    env._object_forces = torch.where(force_fire, new_force, env._object_forces)
    env._object_torques = torch.where(torque_fire, new_torque, env._object_torques)

    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.float().view(-1, 1, 1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.float().view(-1, 1, 1)

    env.scene["object"].write_external_wrench_to_sim(
        forces=env._object_forces,
        torques=env._object_torques,
    )

__all__ = ["apply_action_pipeline", "apply_wrench_dr", "sample_log_uniform"]
