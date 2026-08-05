"""Value-function probe: does the critic assign HIGHER V(s) to better states?

`Loss/value_function` only tells us the critic is *accurate*, not whether its
value landscape is *useful*. This probes that directly: collect states of known
relative quality (fingertip near/far from object, object lifted or not, keypoints
near/far from goal) and check the predicted V(s) increases with task progress.

  Healthy critic  -> V rises monotonically: close>med>far, lifted>not, near>far.
  Useless critic  -> V ~flat/random across buckets (no gradient to guide policy)
                     => explains stagnation even with low value LOSS.

Two rollouts:
  (A) sustained-random actions — guarantees coverage of good states (reaching
      works ~30% of the time this way), so buckets aren't empty.
  (B) the trained policy — reports which buckets the policy actually VISITS.

Run:
  cd mjlab && uv run python -m simtoolreal_g1.deploy.probe_value \
    --checkpoint logs/rsl_rl/simtoolreal_g1/<run>/model_1479.pt
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
    num_envs: int = 512
    steps: int = 150
    device: str = "cuda:0"
    topology: str = "box_head"


def _collect(env, base_env, runner, act_fn, steps):
    """Roll out `steps`, return flat tensors (V, ft_dist, lifted, kp_dist)."""
    Vs, fts, lif, kps = [], [], [], []
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            V = runner.alg.policy.evaluate(obs).squeeze(-1)          # (N,)
            ft = base_env._curr_fingertip_distances.min(dim=-1).values  # (N,)
            Vs.append(V.clone())
            fts.append(ft.clone())
            lif.append(base_env._lifted_object.clone())
            kps.append(base_env._keypoints_max_dist.clone())
            obs, _, _, _ = env.step(act_fn(obs))
    return (torch.cat(Vs), torch.cat(fts), torch.cat(lif), torch.cat(kps))


def _stat(V, mask):
    m = mask
    n = int(m.sum())
    if n == 0:
        return "   n=0        (no samples)"
    v = V[m]
    return f"n={n:>7d}   V mean={v.mean().item():+.4f}  std={v.std().item():.4f}"


def main(args: Args) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    cfg = EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.device = args.device
    base_env = SimToolRealG1Env(cfg, device=args.device, topology=args.topology)
    env = SimToolRealVecEnv(base_env, clip_actions=1.0)

    runner = OnPolicyRunner(
        env, dataclasses.asdict(get_rsl_rl_cfg()), log_dir=None, device=args.device
    )
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    print(f"[INFO] loaded {args.checkpoint}\n")

    n_act = cfg.action.num_actions
    # (A) sustained random per env -> guarantees far->close->lifted coverage.
    sustained = torch.rand(args.num_envs, n_act, device=args.device) * 2 - 1
    V, ft, lifted, kp = _collect(
        env, base_env, runner, lambda o: sustained, args.steps
    )

    print("=" * 68)
    print("VALUE LANDSCAPE  (sustained-random rollout, for coverage)")
    print("=" * 68)
    print("By fingertip->object distance (closer should be HIGHER):")
    print(f"  far   (>0.15m) : {_stat(V, ft > 0.15)}")
    print(f"  med   (.05-.15): {_stat(V, (ft > 0.05) & (ft <= 0.15))}")
    print(f"  close (<0.05m) : {_stat(V, ft <= 0.05)}")
    print("\nBy lifted (lifted should be HIGHER):")
    print(f"  not lifted     : {_stat(V, ~lifted)}")
    print(f"  lifted         : {_stat(V, lifted)}")
    print("\nAmong LIFTED, by keypoint->goal distance (near should be HIGHER):")
    print(f"  far  (>=0.10m) : {_stat(V, lifted & (kp >= 0.10))}")
    print(f"  near (<0.10m)  : {_stat(V, lifted & (kp < 0.10))}")

    # Monotonicity verdict
    def mv(mask):
        return V[mask].mean().item() if int(mask.sum()) > 0 else float("nan")
    v_far, v_med, v_close = mv(ft > 0.15), mv((ft > 0.05) & (ft <= 0.15)), mv(ft <= 0.05)
    v_nl, v_l = mv(~lifted), mv(lifted)
    print("\n" + "-" * 68)
    print("VERDICT:")
    ok_dist = v_close > v_med > v_far
    print(f"  distance monotonic (close>med>far)? {ok_dist}  "
          f"[{v_far:+.3f} -> {v_med:+.3f} -> {v_close:+.3f}]")
    ok_lift = v_l > v_nl
    print(f"  lifted>not-lifted? {ok_lift}  [{v_nl:+.3f} vs {v_l:+.3f}]")

    # (B) what states does the trained policy actually visit?
    _, ftp, lifp, _ = _collect(env, base_env, runner, lambda o: policy(o), args.steps)
    tot = ftp.numel()
    print("\n" + "=" * 68)
    print("POLICY VISITATION  (where the trained policy actually goes)")
    print("=" * 68)
    print(f"  fingertip close (<0.05m): {100*(ftp<=0.05).float().mean():.1f}%")
    print(f"  fingertip med           : {100*((ftp>0.05)&(ftp<=0.15)).float().mean():.1f}%")
    print(f"  fingertip far  (>0.15m) : {100*(ftp>0.15).float().mean():.1f}%")
    print(f"  lifted                  : {100*lifp.float().mean():.1f}%")


if __name__ == "__main__":
    main(tyro.cli(Args))
