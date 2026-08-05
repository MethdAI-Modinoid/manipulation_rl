"""Train SimToolReal-on-G1 with rsl_rl PPO.

Usage:
  uv run python -m simtoolreal_g1.train.train --num-envs 4096 --max-iterations 5000
  uv run python -m simtoolreal_g1.train.train --num-envs 8 --max-iterations 2  # smoke test
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from ..task.env_cfg import EnvCfg
from ..task.multi_object_env import MultiObjectEnv
from ..task.simtoolreal_env import SimToolRealG1Env
from .rl_cfg import get_rsl_rl_cfg
from .vecenv_wrapper import SimToolRealVecEnv


@dataclasses.dataclass(frozen=True)
class Args:
    num_envs: int = 4096
    max_iterations: int = 5000
    num_steps_per_env: int = 16
    device: str = "cuda:0"
    seed: int = 0
    logger: str = "tensorboard"
    wandb_project: str = "simtoolreal_g1"
    view: bool = False
    view_every: int = 2
    multi_tool: bool = True
    tool_split: str = "5,6,1"
    experiment_name: str = "simtoolreal_g1"
    run_name: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


    std_schedule: bool = True
    explore_std: float = 1.0
    exploit_std: float = 0.1
    explore_frac: float = 0.2
    decay_frac: float = 0.3
    std_update_interval: int = 10

def main(args: Args) -> None:
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.device = args.device
    if args.multi_tool:
        split = tuple(int(x) for x in args.tool_split.split(","))
        base_env = MultiObjectEnv(env_cfg, device=args.device, split=split)
        print(f"[INFO] Multi-tool: topologies={base_env.topologies} "
              f"split={base_env.sub_counts} (total {base_env.num_envs})")
    else:
        base_env = SimToolRealG1Env(env_cfg, device=args.device)
    env = SimToolRealVecEnv(
        base_env, clip_actions=1.0, view=args.view, view_every=args.view_every
    )

    agent_cfg = get_rsl_rl_cfg(
        max_iterations=args.max_iterations,
        num_steps_per_env=args.num_steps_per_env,
        experiment_name=args.experiment_name,
        logger=args.logger,
        wandb_project=args.wandb_project,
    )
    agent_cfg = dataclasses.replace(
        agent_cfg,
        seed=args.seed,
        run_name=args.run_name,
        resume=args.resume,
        load_run=args.load_run,
        load_checkpoint=args.load_checkpoint,
    )

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        stamp += f"_{agent_cfg.run_name}"
    log_dir = log_root / stamp
    print(f"[INFO] Logging to: {log_dir}")

    runner = OnPolicyRunner(
        env, dataclasses.asdict(agent_cfg), str(log_dir), device=args.device
    )
    _log_value_function(runner)

    if agent_cfg.resume:
        from mjlab.utils.os import get_checkpoint_path

        resume_path = get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO] Resuming from: {resume_path}")
        runner.load(str(resume_path))

    total = agent_cfg.max_iterations
    if not args.std_schedule:
        runner.learn(num_learning_iterations=total, init_at_random_ep_len=True)
    else:
        _learn_with_std_schedule(runner, args, total)


def _log_value_function(runner) -> None:
    alg = runner.alg
    _orig_update = alg.update

    def _update():
        storage = getattr(alg, "storage", None)
        v = getattr(storage, "values", None) if storage is not None else None
        r = getattr(storage, "returns", None) if storage is not None else None
        vmean = float(v.mean().item()) if v is not None else 0.0
        rmean = float(r.mean().item()) if r is not None else 0.0
        loss_dict = _orig_update()
        loss_dict["value_mean"] = vmean
        loss_dict["return_mean"] = rmean
        return loss_dict

    alg.update = _update


def _set_noise_std(policy, value: float) -> None:
    with torch.no_grad():
        if hasattr(policy, "std"):  # noise_std_type="scalar"
            policy.std.requires_grad_(False)
            policy.std.fill_(value)
        elif hasattr(policy, "log_std"):  # noise_std_type="log"
            policy.log_std.requires_grad_(False)
            policy.log_std.fill_(float(torch.log(torch.tensor(value))))


def _scheduled_std(it: int, total: int, args: "Args") -> float:
    """High std for an explore warmup, then linear decay to a low exploit std."""
    explore_iters = args.explore_frac * total
    decay_iters = max(1.0, args.decay_frac * total)
    if it < explore_iters:
        return args.explore_std
    t = it - explore_iters
    if t < decay_iters:
        frac = t / decay_iters
        return args.explore_std + frac * (args.exploit_std - args.explore_std)
    return args.exploit_std


def _patch_global_progress(runner, total: int, start: int) -> None:
    """Make rsl_rl's per-iteration log show GLOBAL current/total + ETA.

    The std schedule drives training in chunks (runner.learn called repeatedly),
    so rsl_rl's native 'iteration it/tot_iter' shows the chunk end as the total
    and computes a per-chunk ETA. We wrap runner.log to rewrite tot_iter /
    start_iter / num_learning_iterations to their global values, so the console
    reads 'Learning iteration <cur>/<total>' with a correct whole-run ETA."""
    orig_log = runner.log

    def patched_log(locs, width: int = 80, pad: int = 35):
        locs = dict(locs)
        locs["tot_iter"] = total
        locs["start_iter"] = start
        locs["num_learning_iterations"] = total - start
        return orig_log(locs, width, pad)

    runner.log = patched_log


def _learn_with_std_schedule(runner, args: "Args", total: int) -> None:
    """Drive training in chunks, setting the frozen noise std each chunk."""
    policy = runner.alg.policy
    it = runner.current_learning_iteration
    _patch_global_progress(runner, total, it)
    first = True
    while it < total:
        n = min(args.std_update_interval, total - it)
        std = _scheduled_std(it, total, args)
        _set_noise_std(policy, std)
        phase = (
            "EXPLORE" if std >= args.explore_std - 1e-6
            else ("EXPLOIT" if std <= args.exploit_std + 1e-6 else "decay")
        )
        print(f"[std schedule] iter {it}/{total}  noise_std={std:.3f}  ({phase})")
        runner.current_learning_iteration = it  # force correct chunk start
        runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
        first = False
        it += n


if __name__ == "__main__":
    main(tyro.cli(Args))
