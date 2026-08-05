"""
  ./scripts/play.sh --checkpoint <path/to/..._ep_XXXX_rew_YY.pth> \
      --num-envs 1 --render --device cpu

Headless:
  ./scripts/play.sh --checkpoint <path> --num-envs 64 --steps 600 --device cpu
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

if any(a == "--video" or a.startswith("--video=") for a in sys.argv):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import tyro
import yaml

_RLG_PATH = str(Path(__file__).resolve().parents[3] / "rl_games")
if _RLG_PATH not in sys.path:
    sys.path.insert(0, _RLG_PATH)

from .vecenv_adapter import SimToolRealRlGamesVecEnv, load_env_cfg  # noqa: E402

_TRAIN_NUM_ACTORS = 3072
_TRAIN_BLOCK_SIZE = 512
_LEADER_COEF = 50.0  

@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: str
    num_envs: int = 1
    device: str = "cpu"
    render: bool = False
    video: str = ""
    steps: int = 2000
    multi_tool: bool = False
    tool_split: str = "5,6,1"
    deterministic: bool = True
    alpha: float = 1.0

def _build_player(params: dict, vec_env):
    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(params)          # deepcopies params — must NOT contain the env
    runner.set_vec_env(vec_env) 
    player = runner.create_player()
    player.has_batch_dimension = True  # we feed batched (N, obs) observations
    return player

def main(args: Args) -> None:
    import gym

    snap_cfg = None
    if args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent.parent  # <run>/nn/ckpt.pth
        snap_cfg = load_env_cfg(run_dir)
        if snap_cfg is not None:
            print(f"[INFO] loaded training env config snapshot from {run_dir}/env_cfg.pkl")
        else:
            print(f"[WARN] no env_cfg.pkl in {run_dir} — using LIVE env_cfg.py "
                  f"(scene may not match this checkpoint!)")

    if snap_cfg is not None:
        snap_cfg.reset.eval_pose_alpha = args.alpha
        print(f"[INFO] eval_pose_alpha={args.alpha} "
              f"({'full reach-then-grasp' if args.alpha >= 0.999 else 'partial/bootstrap'})")
    adapter = SimToolRealRlGamesVecEnv(
        num_envs=args.num_envs, device=args.device,
        multi_tool=args.multi_tool, tool_split=args.tool_split,
        env_cfg=snap_cfg,
    )
    base = adapter.env  
    base.reset()
    obs_dim = adapter.obs_dim
    state_dim = adapter.state_dim
    n_act = adapter.num_actions

    params = yaml.safe_load(open(Path(__file__).parent / "simtoolreal_sapg.yaml"))
    c = params["params"]["config"]
    c["num_actors"] = _TRAIN_NUM_ACTORS
    c["expl_coef_block_size"] = _TRAIN_BLOCK_SIZE
    c["device"] = args.device
    c["device_name"] = args.device
    c["env_info"] = {
        "observation_space": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        "action_space": gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32),
        "state_space": gym.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32),
        "agents": 1,
        "value_size": 1,
    }
    c["player"] = {
        "deterministic": args.deterministic,
        "games_num": 1,
        "print_stats": False,
    }

    player = _build_player(params, adapter)
    player.restore(args.checkpoint)
    player.reset()  # init LSTM hidden state
    print(f"[INFO] loaded {args.checkpoint}")

    dev = torch.device(args.device)
    coef = torch.full((args.num_envs, 1), _LEADER_COEF, device=dev)

    viewer = mjd = renderer = cam = None
    frames = []
    if args.render or args.video:
        import mujoco

        mjd = base.sim.mj_data
        import numpy as _np

        def _to_np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            try:
                t = x[:]
                if torch.is_tensor(t):
                    return t.detach().cpu().numpy()
            except Exception:
                pass
            if hasattr(x, "numpy"):
                return x.numpy()
            return _np.asarray(x)

        _gs, _gp = base.sim.model.geom_size, base.sim.model.geom_pos
        base.sim.mj_model.geom_size[:] = _to_np(_gs)[0]
        base.sim.mj_model.geom_pos[:] = _to_np(_gp)[0]
        if args.video:
            renderer = mujoco.Renderer(base.sim.mj_model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.28, -0.18, 0.72]  # workspace: right hand + table + cube
            cam.distance, cam.azimuth, cam.elevation = 1.1, 150.0, -20.0
            print(f"[INFO] recording offscreen (EGL) -> {args.video}")
        else:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(base.sim.mj_model, mjd)

    ep_rew = torch.zeros(args.num_envs, device=base.device)
    ep_count = 0
    ep_rew_sum = 0.0
    lifted_hits = 0
    success_hits = 0

    for t in range(args.steps):
        actor_obs = base.obs_buf["policy"].to(dev)
        obs = torch.cat([actor_obs, coef], dim=1)  
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=args.deterministic)
        action = action.reshape(args.num_envs, n_act).to(base.device)

        _, rew, term, trunc, _ = base.step(action)
        ep_rew += rew
        done = term | trunc
        lifted_hits += int(base._lifted_object.sum().item())
        success_hits += int(base._successes.sum().item())

        if bool(done.any()):
            ep_count += int(done.sum().item())
            ep_rew_sum += float(ep_rew[done].sum().item())
            ep_rew[done] = 0.0
            player.reset()  

        if viewer is not None or renderer is not None:
            import mujoco

            mjd.qpos[:] = base.sim.data.qpos[0].detach().cpu().numpy()
            mjd.qvel[:] = base.sim.data.qvel[0].detach().cpu().numpy()
            mujoco.mj_forward(base.sim.mj_model, mjd)
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
            else:
                renderer.update_scene(mjd, cam)
                frames.append(renderer.render())

    if viewer is not None:
        viewer.close()
    if renderer is not None:
        import imageio
        renderer.close()
        imageio.mimwrite(args.video, frames, fps=30, quality=8)
        print(f"[INFO] wrote {args.video}  ({len(frames)} frames, {len(frames)/30:.1f}s)")

    total_steps = (t + 1) * args.num_envs
    print(f"[RESULT] steps={t + 1}  episodes={ep_count}")
    print(f"  mean episode reward : {ep_rew_sum / max(1, ep_count):.3f}")
    print(f"  lifted step-fraction: {lifted_hits / max(1, total_steps):.4f}")
    print(f"  success step-hits   : {success_hits}")

if __name__ == "__main__":
    main(tyro.cli(Args))
