from __future__ import annotations

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

def get_rsl_rl_cfg(
    max_iterations: int = 5000,
    num_steps_per_env: int = 16,
    experiment_name: str = "simtoolreal_g1",
    logger: str = "tensorboard",
    wandb_project: str = "simtoolreal_g1",
) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=0,
        num_steps_per_env=num_steps_per_env,  # rl_games horizon_length
        max_iterations=max_iterations,
        save_interval=200,
        experiment_name=experiment_name,
        logger=logger,  # type: ignore[arg-type]
        wandb_project=wandb_project,  # rsl_rl WandbSummaryWriter reads this
        obs_groups={"policy": ("policy",), "critic": ("critic",)},
        clip_actions=1.0,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(1024, 1024, 512, 512),
            critic_hidden_dims=(1024, 1024, 512, 512),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=4,   # rl_games mini_epochs
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.0,
            desired_kl=0.016,
            max_grad_norm=1.0,
            value_loss_coef=4.0,     # rl_games critic_coef
            use_clipped_value_loss=True,
            clip_param=0.1,          # rl_games e_clip
        ),
    )
