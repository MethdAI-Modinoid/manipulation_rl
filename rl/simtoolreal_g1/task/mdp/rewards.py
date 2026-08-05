from __future__ import annotations
import torch

def lifting_reward(
    object_z: torch.Tensor,            # (N,)
    object_init_z: torch.Tensor,       # (N,)
    prev_lifted: torch.Tensor,         # (N,) bool
    lifting_bonus_threshold: float,
    lifting_bonus: float,
    lifting_rew_scale: float,
    fingertip_dist: torch.Tensor,      # (N, 5) fingertip -> object distance
    grasp_gate_std: float,             # gate width (m); ~0 => no gate
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if grasp_gate_std > 0.0:
        gate = torch.exp(-(fingertip_dist ** 2) / (grasp_gate_std ** 2)).mean(dim=-1)
    else:
        gate = torch.ones_like(object_z)
    grasped = gate > 0.5   # fingertips genuinely on the object (mean gate high)
    z_lift = object_z - object_init_z
    lift_rew = torch.clamp(z_lift, 0.0, 0.3) * gate
    lift_bonus = lifting_bonus * (z_lift > lifting_bonus_threshold).float() * gate
    # latch drives the reach/keypoint handoff; grasp-gated so a bat can't flip it
    lifted = ((z_lift > lifting_bonus_threshold) & grasped) | prev_lifted
    return lift_rew * lifting_rew_scale, lift_bonus, lifted

def reach_reward(
        palm_pos: torch.Tensor,    # (N, 3) env-local palm position
        object_pos: torch.Tensor,  # (N, 3) env-local object position
        std: float,
        scale: float,
) -> torch.Tensor:
    d2 = torch.sum((palm_pos - object_pos) ** 2, dim=-1)
    return scale * torch.exp(-d2 / std ** 2)

def distance_delta_reward(
        curr_fingertip_dist: torch.Tensor,
        closest_fingertip_dist: torch.Tensor,
        lifted: torch.Tensor,
        rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ratchet: pays PROGRESS toward the object, mean over fingers, capped anti-spike
    deltas = torch.clamp(closest_fingertip_dist - curr_fingertip_dist, 0.0, 0.05)
    new_closest = torch.minimum(closest_fingertip_dist, curr_fingertip_dist)
    rew = deltas.mean(dim=-1)
    return rew * rew_scale, new_closest

def keypoint_reward(
    keypoints_max_dist: torch.Tensor,
    closest_keypoint_max_dist: torch.Tensor,
    lifted: torch.Tensor,
    rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = closest_keypoint_max_dist - keypoints_max_dist
    new_closest = torch.minimum(closest_keypoint_max_dist, keypoints_max_dist)
    delta = torch.clamp(delta, 0.0, 100.0)
    rew = delta * lifted.float()
    return rew * rew_scale, new_closest

def action_penalty(
        joint_vel: torch.Tensor,
        arm_ids,
        hand_ids,
        arm_scale: float,
        hand_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    arm = -arm_scale * joint_vel[:, arm_ids].abs().sum(dim=-1)
    hand = -hand_scale * joint_vel[:, hand_ids].abs().sum(dim=-1)
    return arm, hand


def bring_to_goal_reward(
        object_pos: torch.Tensor,   # (N, 3) env-local
        goal_pos: torch.Tensor,     # (N, 3) env-local
        lifted: torch.Tensor,       # (N,) bool
        std: float,
        scale: float,
) -> torch.Tensor:
    d2 = torch.sum((object_pos - goal_pos) ** 2, dim=-1)
    return scale * torch.exp(-d2 / (std ** 2)) * lifted.float()


def joint_pos_limits_penalty(
        joint_pos: torch.Tensor,     # (N, J)
        soft_limits: torch.Tensor,   # (N, J, 2) [lower, upper]
        scale: float,
) -> torch.Tensor:
    lower = soft_limits[..., 0]
    upper = soft_limits[..., 1]
    below = (lower - joint_pos).clamp(min=0.0)
    above = (joint_pos - upper).clamp(min=0.0)
    return -scale * (below + above).sum(dim=-1)


def premature_curl_penalty(
        hand_joint_pos: torch.Tensor,   # (N, n_hand) finger joint angles (0 = open, + = curled)
        palm_pos: torch.Tensor,         # (N, 3) env-local
        object_pos: torch.Tensor,       # (N, 3) env-local
        near_radius: float,             # curl freely once palm is within this of the cube
        scale: float,
) -> torch.Tensor:
    flexion = hand_joint_pos.clamp(min=0.0).sum(dim=-1)          # total finger curl
    palm_dist = torch.norm(palm_pos - object_pos, dim=-1)
    far = (palm_dist - near_radius).clamp(min=0.0)               # 0 when at the cube
    return -scale * flexion * far

def workspace_penalty(
        palm_pos: torch.Tensor,   # (N, 3) env-local
        box_min: torch.Tensor,    # (3,)
        box_max: torch.Tensor,    # (3,)
        scale: float,
) -> torch.Tensor:
    below = (box_min - palm_pos).clamp(min=0.0)
    above = (palm_pos - box_max).clamp(min=0.0)
    exceed = (below + above).sum(dim=-1)
    return -scale * exceed


def object_shove_penalty(
        obj_lin_vel_w: torch.Tensor,   # (N, 3) object world linear velocity
        lifted: torch.Tensor,          # (N,) bool
        scale: float,
        deadzone: float = 0.0,
) -> torch.Tensor:
    # penalizes the object's FULL 3D speed, always active (discourages batting/flinging)
    return -scale * torch.norm(obj_lin_vel_w, dim=-1)


def update_near_goal_steps(
        near_goal: torch.Tensor,
        near_goal_steps: torch.Tensor,
        force_consecutive: bool,
) -> torch.Tensor:
    ng = near_goal.long()
    if force_consecutive:
        return (near_goal_steps + ng) * ng
    return near_goal_steps + ng

def reach_goal_bonus(
        near_goal: torch.Tensor,
        is_success: torch.Tensor,
        reach_goal_bonus_value: float,
        success_steps: int,
        force_consecutive: bool,
) -> torch.Tensor:
    if force_consecutive:
        return is_success.float() * reach_goal_bonus_value
    return near_goal.float() * (reach_goal_bonus_value / success_steps)

def grasp_reward(
    fingertip_dist: torch.Tensor,  # (N, 5) per-fingertip distance to object
    grasp_radius: float,
    rew_scale: float,
) -> torch.Tensor:
    # per-fingertip proximity to the object CENTER: 1.0 touching -> 0.0 at grasp_radius
    closeness = (1.0 - fingertip_dist / grasp_radius).clamp(0.0, 1.0)
    return rew_scale * closeness.sum(dim=-1)

def grasp_contact_reward(
        n_fingers_contact: torch.Tensor,  # (N,) count of fingertips touching object
        scale: float,
        lifted_gate: torch.Tensor | None = None,  # (N,) 1.0 while cube is off the table
) -> torch.Tensor:
    # reads real sim contacts, not proximity. Gating on lifted_gate pays for grip
    # RETENTION during the lift and zero for gripping a resting cube (unfarmable).
    r = scale * n_fingers_contact
    if lifted_gate is not None:
        r = r * lifted_gate
    return r

def grasp_contact_ratchet(n_fingers_contact, max_fingers_contact, scale):
    # pays once per NEW fingertip contact this episode, so a steady grip earns nothing
    # more and the policy keeps adding fingers. Returns (reward, new_max).
    new_max = torch.maximum(max_fingers_contact, n_fingers_contact)
    gain = (new_max - max_fingers_contact).clamp_min(0.0)  # only NEW fingers pay
    return scale * gain, new_max

def compute_rewards(env) -> torch.Tensor:
    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    obj_pos = env.scene["object"].data.root_link_pos_w - env_origins

    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=obj_pos[:, 2],
        object_init_z=env._object_init_z,
        prev_lifted=env._lifted_object,
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
        fingertip_dist=env._curr_fingertip_distances,
        grasp_gate_std=rew_cfg.grasp_gate_std,
    )
    env._lifted_object = new_lifted

    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=env._closest_fingertip_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.distance_delta_rew_scale,
    )
    env._closest_fingertip_dist = new_closest_ft

    reach_rew = reach_reward(
        palm_pos=env._palm_pos,
        object_pos=obj_pos,
        std=rew_cfg.reach_rew_std,
        scale=rew_cfg.reach_rew_scale,
    )

    kp_rew, new_closest_kp = keypoint_reward(
        keypoints_max_dist=env._keypoints_max_dist,
        closest_keypoint_max_dist=env._closest_keypoint_max_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.keypoint_rew_scale,
    )
    env._closest_keypoint_max_dist = new_closest_kp

    arm_pen, hand_pen = action_penalty(
        joint_vel=env.scene["robot"].data.joint_vel,
        arm_ids=env._arm_joint_ids,
        hand_ids=env._hand_joint_ids,
        arm_scale=rew_cfg.arm_actions_penalty_scale,
        hand_scale=rew_cfg.hand_actions_penalty_scale,
    )

    curl_pen = premature_curl_penalty(
        hand_joint_pos=env.scene["robot"].data.joint_pos[:, env._hand_joint_ids],
        palm_pos=env._palm_pos,
        object_pos=obj_pos,
        near_radius=rew_cfg.curl_near_radius,
        scale=rew_cfg.premature_curl_penalty_scale,
    )

    jlim_pen = joint_pos_limits_penalty(
        joint_pos=env.scene["robot"].data.joint_pos,
        soft_limits=env.scene["robot"].data.soft_joint_pos_limits,
        scale=rew_cfg.joint_pos_limit_penalty_scale,
    )

    bring_rew = bring_to_goal_reward(
        object_pos=obj_pos,
        goal_pos=env._goal_pos,
        lifted=env._lifted_object,
        std=rew_cfg.bring_to_goal_std,
        scale=rew_cfg.bring_to_goal_scale,
    )

    bonus = reach_goal_bonus(
        near_goal=env._near_goal,
        is_success=env._is_success,
        reach_goal_bonus_value=rew_cfg.reach_goal_bonus,
        success_steps=term_cfg.success_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )

    ws_pen = workspace_penalty(
        palm_pos=env._palm_pos,
        box_min=env._workspace_box_min,
        box_max=env._workspace_box_max,
        scale=rew_cfg.workspace_penalty_scale,
    )

    grasp_rew = grasp_reward(
        fingertip_dist=env._curr_fingertip_distances,
        grasp_radius=rew_cfg.grasp_radius,
        rew_scale=rew_cfg.grasp_rew_scale,
    )

    shove_pen = object_shove_penalty(
        obj_lin_vel_w=env.scene["object"].data.root_link_lin_vel_w,
        lifted=env._lifted_object,
        scale=rew_cfg.shove_penalty_scale,
        deadzone=rew_cfg.shove_penalty_deadzone,
    )

    grasp_ratchet, new_max_fc = grasp_contact_ratchet(
        n_fingers_contact=env._n_fingers_contact,
        max_fingers_contact=env._max_fingers_contact,
        scale=rew_cfg.grasp_ratchet_scale,
    )
    env._max_fingers_contact = new_max_fc
    # closure ratchet + small lifted-gated retention term (keep fingers on through the lift)
    lifted_now = ((obj_pos[:, 2] - env._object_init_z) > 0.03).float()
    grasp_contact = grasp_contact_reward(
        n_fingers_contact=env._n_fingers_contact,
        scale=rew_cfg.grasp_contact_scale,
        lifted_gate=lifted_now,
    ) + grasp_ratchet

    reward = (
        lift_rew + lift_bonus_rew + ft_rew + reach_rew + kp_rew + arm_pen + hand_pen
        + bonus + ws_pen + grasp_rew + shove_pen + grasp_contact + curl_pen + jlim_pen
        + bring_rew
    )
    # global reward_shaper scale (simtoolreal rl_games scale_value: 0.01)
    reward = reward * rew_cfg.reward_shaper_scale
    env._reward_terms = {
        "fingertip_delta_rew": ft_rew,
        "reach_rew": reach_rew,
        "lifting_rew": lift_rew,
        "lift_bonus_rew": lift_bonus_rew,
        "keypoint_rew": kp_rew,
        "arm_actions_penalty": arm_pen,
        "hand_actions_penalty": hand_pen,
        "bonus_rew": bonus,
        "workspace_penalty": ws_pen,
        "grasp_rew": grasp_rew,
        "shove_penalty": shove_pen,
        "grasp_contact_rew": grasp_contact,
        "bring_rew": bring_rew,
        "jlim_penalty": jlim_pen,
        "total_reward": reward,
    }
    return reward

__all__ = [
    "lifting_reward",
    "distance_delta_reward",
    "keypoint_reward",
    "action_penalty",
    "workspace_penalty",
    "grasp_reward",
    "grasp_contact_reward",
    "object_shove_penalty",
    "update_near_goal_steps",
    "reach_goal_bonus",
    "compute_rewards",
]
