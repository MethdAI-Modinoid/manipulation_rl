"""Roll out a trained SimToolReal-on-G1 policy in sim (sim2sim eval).

Loads an rsl_rl checkpoint, runs the deterministic inference policy in the same
MuJoCo-Warp env used for training, and reports task metrics (reward, lifted
fraction, success rate, current keypoint tolerance). With --render it also opens
a passive MuJoCo viewer following env 0.

Usage:
  # headless metrics over 600 steps on 64 envs
  uv run python -m simtoolreal_g1.deploy.run_policy --checkpoint logs/rsl_rl/simtoolreal_g1/<run>/model_5000.pt

  # watch a single env live
  uv run python -m simtoolreal_g1.deploy.run_policy --checkpoint <path> --num-envs 1 --render
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from ..task.env_cfg import EnvCfg
from ..task.simtoolreal_env import SimToolRealG1Env
from ..train.rl_cfg import get_rsl_rl_cfg
from ..train.vecenv_wrapper import SimToolRealVecEnv


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    """Path to the rsl_rl .pt checkpoint."""
    num_envs: int = 64
    num_steps: int = 600
    device: str = "cuda:0"
    render: bool = False
    """Open a passive MuJoCo viewer following env 0 (needs a display)."""
    eval_tolerance: float | None = None
    """Fix keypoint success tolerance for eval (e.g. 0.01). None = env default."""


def main(args: Args) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.device = args.device
    if args.eval_tolerance is not None:
        env_cfg.termination.eval_success_tolerance = args.eval_tolerance

    base_env = SimToolRealG1Env(env_cfg, device=args.device)
    env = SimToolRealVecEnv(base_env, clip_actions=1.0)

    agent_cfg = get_rsl_rl_cfg()
    runner = OnPolicyRunner(
        env, dataclasses.asdict(agent_cfg), log_dir=None, device=args.device
    )
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    viewer = mjd = None
    if args.render:
        import mujoco
        import mujoco.viewer

        mjd = base_env.sim.mj_data
        viewer = mujoco.viewer.launch_passive(base_env.sim.mj_model, mjd)

    obs = env.get_observations()
    rew_sum = torch.zeros(args.num_envs, device=args.device)
    n_steps = 0
    ep_success_rates: list[float] = []

    with torch.inference_mode():
        for step in range(args.num_steps):
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
            rew_sum += rew
            n_steps += 1

            # Episodes that just ended: log their success outcome.
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                succ = (base_env._prev_episode_successes[done_ids] > 0).float()
                ep_success_rates.extend(succ.cpu().tolist())

            if args.render and viewer is not None:
                import mujoco

                mjd.qpos[:] = base_env.sim.data.qpos[0].cpu().numpy()
                mjd.qvel[:] = base_env.sim.data.qvel[0].cpu().numpy()
                mujoco.mj_forward(base_env.sim.mj_model, mjd)
                viewer.sync()
                if not viewer.is_running():
                    break

            if step % 50 == 0 or step == args.num_steps - 1:
                lifted = base_env._lifted_object.float().mean().item()
                tol = base_env._current_success_tolerance
                cur_succ = base_env._successes.float().mean().item()
                print(
                    f"step {step:4d} | rew/step {rew.mean().item():+.3f} "
                    f"| lifted {lifted:5.1%} | mean_successes {cur_succ:.2f} "
                    f"| tol {tol:.4f}"
                )

    print("\n=== Rollout summary ===")
    print(f"reward (sum over {n_steps} steps, mean across envs): {rew_sum.mean().item():.3f}")
    print(f"final lifted fraction: {base_env._lifted_object.float().mean().item():.1%}")
    if ep_success_rates:
        sr = sum(ep_success_rates) / len(ep_success_rates)
        print(f"episode success rate: {sr:.1%}  (over {len(ep_success_rates)} finished eps)")
    else:
        print("episode success rate: n/a (no episodes finished in window)")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
