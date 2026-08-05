from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class SceneCfg:
    num_envs: int = 4096
    env_spacing: float = 2.5

@dataclass
class ObjectCfg:
    handle_head_types: tuple[str, ...] = (
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    )
    num_assets_per_type: int = 100
    object_base_size: float = 0.04
    object_friction: float = 0.5
    fingertip_friction: float = 1.5

    bootstrap_grasp: bool = True
    bootstrap_handle_full: tuple[float, float, float] = (0.06, 0.06, 0.06)
    bootstrap_density: float = 185.0   # ~0.04 kg for a 6 cm cube (their mass)
    bootstrap_friction: float = 3.0    # grippy (matches their cube_geom friction)

    object_min_inertia: float = 1e-4
    topologies: tuple[str, ...] = ("box_only",)
    topology_split: tuple[int, ...] = (1,)

@dataclass
class ObsCfg:
    state_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "palm_vel",
        "object_rot",
        "object_vel",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist",
        "closest_fingertip_dist",
        "lifted_object",
        "progress",
        "successes",
        "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
    )
    clamp_abs_observations: float = 10.0

@dataclass
class ActionCfg:
    num_actions: int = 13  # right arm (7) + right hand (6)
    arm_moving_average: float = 0.4
    hand_moving_average: float = 0.8
    dof_speed_scale: float = 1.5

@dataclass
class RewardCfg:
    reward_shaper_scale: float = 1.0      

    keypoint_rew_scale: float = 0.0        
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size: tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    fixed_size_keypoint_reward: bool = True

    bring_to_goal_scale: float = 20.0      
    bring_to_goal_std: float = 0.1
    lifting_rew_scale: float = 20.0        
    lifting_bonus: float = 5.0            
    grasp_gate_std: float = 0.08
    lifting_bonus_threshold: float = 0.05   
    distance_delta_rew_scale: float = 30.0 
    reach_rew_scale: float = 1.0            
    reach_rew_std: float = 0.12
    reach_goal_bonus: float = 0.0   

    arm_actions_penalty_scale: float = 0.005
    hand_actions_penalty_scale: float = 0.003
    grasp_rew_scale: float = 0.0
    grasp_radius: float = 0.08
    workspace_penalty_scale: float = 0.0
    workspace_box_min: tuple[float, float, float] = (-0.05, -0.40, 0.55)
    workspace_box_max: tuple[float, float, float] = (0.45, 0.15, 1.25)
    shove_penalty_scale: float = 0.5
    shove_penalty_deadzone: float = 0.4

    grasp_contact_scale: float = 0.0
    grasp_ratchet_scale: float = 0.0

    premature_curl_penalty_scale: float = 0.0
    curl_near_radius: float = 0.08
    joint_pos_limit_penalty_scale: float = 1.0  

@dataclass
class ResetCfg:
    # FULL cube scatter (m) around the table center; actually applied × alpha, so
    # scatter grows 0→this as the start-pose anneals from b1-over-cube to neutral.
    reset_position_noise_x: float = 0.06
    reset_position_noise_y: float = 0.06
    reset_position_noise_z: float = 0.0

    # Radian noise on the arm/finger start pose (× alpha: 0 early, full late).
    reset_dof_pos_random_interval_arm: float = 0.08
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.0

    object_start_pos: tuple[float, float, float] = (0.24, -0.19, 0.69)
    object_start_z_range: float = 0.02
    table_pos: tuple[float, float, float] = (0.24, -0.19, 0.64)  # top 0.66 (half 0.02)
    table_half_extents: tuple[float, float, float] = (0.15, 0.15, 0.02)
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (0.10, -0.30, 0.92)
    target_volume_maxs: tuple[float, float, float] = (0.40, 0.00, 1.12)
    target_volume_region_scale: float = 1.0

    object_upright_spawn: bool = True
    object_yaw_range: float = 0.7
    easy_goal: bool = True
    easy_goal_lift: float = 0.15
    goal_sampling_type: str = "absolute"

    eval_pose_alpha: float | None = None   # Phase 2: ramp ENABLED (alpha anneals 0->1 over training). Set to 0.0 to re-freeze at fixed b1.
    grasp_arm_pose: tuple[float, ...] = (-0.5, -0.2, 0.0, 0.8, 0.0, 1.2, 0.0)   # anneal START (over cube)
    home_arm_pose: tuple[float, ...] = (-0.3, -0.25, 0.0, 0.6, 0.0, 0.0, 0.0)   # anneal END (hover above
    pose_anneal_frames: int = 150000

@dataclass
class TerminationCfg:
    episode_length: int = 1200 

    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None

    success_steps: int = 10
    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False

    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    tolerance_curriculum_success_threshold: float = 3.0

@dataclass
class DomainRandomizationCfg:
    use_obs_delay: bool = False
    obs_delay_max: int = 3
    use_action_delay: bool = False
    action_delay_max: int = 3

    use_object_state_delay_noise: bool = False
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)

    joint_velocity_obs_noise_std: float = 0.0   # DR off for bootstrap (see above)

    force_scale: float = 0.0   # DR off for bootstrap: no random wrenches during grasp
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True

    torque_scale: float = 0.0   # DR off for bootstrap
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True
    object_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    fingertip_friction_scale_range: tuple[float, float] = (1.0, 1.0)


@dataclass
class EnvCfg:

    decimation: int = 4
    sim_dt: float = 0.005  # 200 Hz physics / 50 Hz policy 
    device: str = "cuda:0"
    njmax: int = 1024
    nconmax: int = 256

    scene: SceneCfg = field(default_factory=SceneCfg)
    object: ObjectCfg = field(default_factory=ObjectCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    domain_randomization: DomainRandomizationCfg = field(
        default_factory=DomainRandomizationCfg
    )

__all__ = [
    "SceneCfg",
    "ObjectCfg",
    "ObsCfg",
    "ActionCfg",
    "RewardCfg",
    "ResetCfg",
    "TerminationCfg",
    "DomainRandomizationCfg",
    "EnvCfg",
]
