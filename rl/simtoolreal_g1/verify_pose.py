from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import tyro
import dataclasses

_RLG = str(Path(__file__).resolve().parents[2] / "rl_games")
if _RLG not in sys.path:
    sys.path.insert(0, _RLG)

from .rlg.vecenv_adapter import SimToolRealRlGamesVecEnv  # noqa: E402

@dataclasses.dataclass(frozen=True)
class Args:
    steps: int = 4000
    device: str = "cpu"
    close: bool = False
    alpha: float = 0.0

def main(a: Args) -> None:
    import mujoco
    import mujoco.viewer

    ad = SimToolRealRlGamesVecEnv(num_envs=1, device=a.device)
    base = ad.env
    base.cfg.reset.eval_pose_alpha = a.alpha  # freeze the start-pose alpha for eval
    base.reset()
    n_act = ad.num_actions

    mjd = base.sim.mj_data
    # Sync the runtime (per-world) object geom size/pos into the viewer model, else
    # the viewer draws the flat spec placeholder instead of the real cube.
    _gs, _gp = base.sim.model.geom_size, base.sim.model.geom_pos
    base.sim.mj_model.geom_size[:] = (
        _gs[0].detach().cpu().numpy() if torch.is_tensor(_gs) else np.asarray(_gs)[0]
    )
    base.sim.mj_model.geom_pos[:] = (
        _gp[0].detach().cpu().numpy() if torch.is_tensor(_gp) else np.asarray(_gp)[0]
    )
    viewer = mujoco.viewer.launch_passive(base.sim.mj_model, mjd)

    for t in range(a.steps):
        act = torch.zeros(1, n_act, device=base.device)
        # arm delta 0 = hold b1; hand -1 open, +1 closed
        act[:, 7:] = min(1.0, t / 120.0) if a.close else -1.0
        base.step(act)
        mjd.qpos[:] = base.sim.data.qpos[0].detach().cpu().numpy()
        mjd.qvel[:] = base.sim.data.qvel[0].detach().cpu().numpy()
        mujoco.mj_forward(base.sim.mj_model, mjd)
        viewer.sync()
        if not viewer.is_running():
            break
    viewer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
