"""
./scripts/train.sh --num-envs 12288 --num-blocks 6 --max-iterations 10000 --wandb
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import datetime
from pathlib import Path
import tyro
import yaml

_RLG_PATH = str(Path(__file__).resolve().parents[3] / "rl_games")
if _RLG_PATH not in sys.path:
    sys.path.insert(0, _RLG_PATH)

from .vecenv_adapter import EnvStatsAlgoObserver, SimToolRealRlGamesVecEnv  # noqa: E402

@dataclasses.dataclass(frozen=True)
class Args:
    num_envs: int = 12288
    num_blocks: int = 6
    max_iterations: int = 10000
    horizon_length: int = 16
    num_minibatches: int = 4
    device: str = "cuda:0"
    seed: int = 42
    multi_tool: bool = False

    tool_split: str = "5,6,1"
    experiment_name: str = "simtoolreal_g1_sapg"
    run_name: str = ""
    train_dir: str = "logs/rlg"
    wandb: bool = False
    wandb_project: str = "simtoolreal_g1"
    checkpoint: str = ""

def _build_params(args: Args, num_actors: int, block_size: int, minibatch: int,
                  full_name: str) -> dict:
    cfg_path = Path(__file__).parent / "simtoolreal_sapg.yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)

    p = params["params"]
    p["seed"] = args.seed
    if args.checkpoint:
        p["load_checkpoint"] = True
        p["load_path"] = args.checkpoint

    c = p["config"]
    c["num_actors"] = num_actors
    c["max_epochs"] = args.max_iterations
    c["horizon_length"] = args.horizon_length
    c["seq_length"] = args.horizon_length
    c["minibatch_size"] = minibatch
    c["central_value_config"]["minibatch_size"] = minibatch
    c["expl_coef_block_size"] = block_size
    c["device"] = args.device
    c["device_name"] = args.device
    c["train_dir"] = args.train_dir
    c["name"] = full_name
    c["full_experiment_name"] = full_name
    return params

def main(args: Args) -> None:
    if args.num_blocks < 2:
        raise ValueError("SAPG needs num_blocks >= 2 (1 leader + >=1 explorer).")

    env = SimToolRealRlGamesVecEnv(
        num_envs=args.num_envs,
        device=args.device,
        multi_tool=args.multi_tool,
        tool_split=args.tool_split,
    )
    num_actors = env.num_envs
    print(f"[INFO] env built: num_actors={num_actors} obs={env.obs_dim} "
          f"state={env.state_dim} act={env.num_actions}")

    if num_actors % args.num_blocks != 0:
        raise ValueError(
            f"num_actors ({num_actors}) not divisible by num_blocks "
            f"({args.num_blocks}). Pick num_envs a multiple of num_blocks."
        )
    block_size = num_actors // args.num_blocks
    if num_actors % args.num_minibatches != 0:
        raise ValueError(
            f"num_actors ({num_actors}) not divisible by num_minibatches "
            f"({args.num_minibatches})."
        )
    minibatch = args.horizon_length * num_actors // args.num_minibatches
    print(f"[INFO] SAPG: {args.num_blocks} blocks x {block_size} envs; "
          f"batch={args.horizon_length * num_actors} minibatch={minibatch}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_name = f"0_{args.experiment_name}"
    if args.run_name:
        full_name += f"_{args.run_name}"
    full_name += f"_{stamp}"

    from .vecenv_adapter import save_env_cfg
    run_dir = Path(args.train_dir) / full_name
    save_env_cfg(env.cfg, run_dir)
    print(f"[INFO] snapshotted env config -> {run_dir}/env_cfg.pkl")

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=full_name,
            sync_tensorboard=True,  
            config={
                "num_envs": num_actors, "num_blocks": args.num_blocks,
                "block_size": block_size, "max_iterations": args.max_iterations,
                "multi_tool": args.multi_tool, "algo": "SAPG",
            },
        )

    params = _build_params(args, num_actors, block_size, minibatch, full_name)

    from rl_games.torch_runner import Runner

    runner = Runner(algo_observer=EnvStatsAlgoObserver())
    runner.load(params)
    runner.reset()
    runner.set_vec_env(env) 

    run_args = {
        "train": True,
        "play": False,
        "checkpoint": args.checkpoint or None,
        "sigma": None,
    }
    runner.run(run_args)


if __name__ == "__main__":
    main(tyro.cli(Args))
