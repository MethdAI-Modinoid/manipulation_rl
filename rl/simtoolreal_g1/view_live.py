"""Interactive NATIVE MuJoCo viewer on the LIVE MuJoCo-Warp env (no training).

Unlike the offscreen OpenCV window used during training (fixed camera, 2-D image
controls only), this drives the real on-screen passive viewer
(``mujoco.viewer.launch_passive``) — full orbit / zoom / pan — following env 0 of
a chosen object topology, stepping physics live so you can watch the arm + hand +
object react.

The GLX/CUDA context clash only bites when the full PPO trainer is running; with
the GPU otherwise free this works (same path as ``run_policy --render``).

Run:
  ./scripts/view.sh view_live --topology cyl_head      # marker / cylinder
  ./scripts/view.sh view_live --topology box_only      # eraser (no head)
  ./scripts/view.sh view_live --topology box_head      # hammer/etc (cuboid)
  ./scripts/view.sh view_live --actions zero           # object just rests
"""
from __future__ import annotations

import dataclasses
import time

import mujoco
import mujoco.viewer
import torch
import tyro

from .task.env_cfg import EnvCfg
from .task.simtoolreal_env import SimToolRealG1Env


@dataclasses.dataclass(frozen=True)
class Args:
    topology: str = "cyl_head"
    """Which object to watch: box_head | cyl_head | box_only."""
    actions: str = "random"
    """'random' (jitter arm+hand so you see motion) or 'zero' (hold rest pose)."""
    num_envs: int = 4
    """A few envs on the GPU; the viewer follows env 0."""
    hold: int = 6
    """Repeat each random action for this many steps (smoother, less jittery)."""
    device: str = "cuda:0"
    seed: int = 0
    realtime: bool = True
    """Pace the loop to ~wall-clock so motion looks natural."""


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    cfg = EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.device = args.device
    env = SimToolRealG1Env(cfg, device=args.device, topology=args.topology)

    mjm = env.sim.mj_model
    mjd = mujoco.MjData(mjm)
    n_act = cfg.action.num_actions
    step_dt = env.step_dt

    print(f"[view_live] topology={args.topology} actions={args.actions} "
          f"num_envs={args.num_envs} — close the window to exit.")

    with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
        viewer.cam.lookat[:] = [0.25, -0.15, 0.95]
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15

        a = torch.zeros(args.num_envs, n_act, device=args.device)
        i = 0
        while viewer.is_running():
            t0 = time.time()
            if args.actions == "zero":
                a = torch.zeros(args.num_envs, n_act, device=args.device)
            elif i % max(1, args.hold) == 0:
                a = torch.rand(args.num_envs, n_act, device=args.device) * 2.0 - 1.0
            env.step(a)
            i += 1

            mjd.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mjd.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mjm, mjd)
            viewer.sync()

            if args.realtime:
                dt = step_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)


if __name__ == "__main__":
    main(tyro.cli(Args))
