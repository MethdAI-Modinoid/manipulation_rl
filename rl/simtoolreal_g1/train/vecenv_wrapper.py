"""rsl_rl VecEnv wrapper for the Direct-style SimToolRealG1Env.

SimToolRealG1Env is not a mjlab ManagerBasedRlEnv, but its step() already
returns the same 5-tuple (obs_dict, reward, terminated, truncated, extras) and
its observations are a dict {"policy": (N,92), "critic": (N,114)} — which maps
directly onto rsl_rl's obs_groups {"policy": ("policy",), "critic": ("critic",)}
for asymmetric actor-critic. This wrapper adapts that env to rsl_rl's VecEnv
interface (get_observations / step returning a TensorDict + combined dones).
"""
from __future__ import annotations

from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from ..task.simtoolreal_env import SimToolRealG1Env


class SimToolRealVecEnv(VecEnv):
    def __init__(
        self,
        env: SimToolRealG1Env,
        clip_actions: float | None = None,
        view: bool = False,
        view_every: int = 2,
    ):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.cfg.action.num_actions

        # rsl_rl does not call reset() itself; ensure a valid initial state.
        self.env.reset()

        # Optional live viewer so you can WATCH training happen (Isaac-Lab
        # style). Renders OFFSCREEN via EGL and blits into an OpenCV window. EGL
        # (unlike the on-screen GLX passive viewer) coexists with MuJoCo-Warp's
        # CUDA context, so this works WHILE training on the GPU — no
        # GLXBadContext. Rendered every `view_every` policy steps for throughput.
        #
        # It tiles up to _VIEW_MAX worlds into ONE scene (mjlab's mjv_addGeoms
        # trick). Our G1 is WELDED (fixed base), so unlike free-base locomotion
        # every world's robot sits at the SAME model coords — we therefore place
        # each world on a synthetic grid ourselves, and undo the object's
        # env_origin offset (the sim bakes it into the object's freejoint qpos)
        # so each object lands next to its own robot.
        self._viewer = None
        self._view_every = max(1, view_every)
        self._view_ct = 0
        if view:
            import math
            import os

            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import cv2
            import mujoco
            import numpy as np

            self._cv2 = cv2
            # Gather ALL sub-envs (box_head/cyl_head/box_only) so the grid shows
            # a MIX of tool shapes, not just box_head. Each sub-env is a separate
            # compiled model, so we keep a model + MjData + origins per sub-env
            # and add each world's geoms via mjv_addGeoms (models can differ).
            subs = list(getattr(env, "sub_envs", [env]))
            self._subs = []
            for se in subs:
                mjm = se.sim.mj_model
                fq = None
                for jid in range(mjm.njnt):
                    if mjm.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                        fq = int(mjm.jnt_qposadr[jid])
                        break
                self._subs.append({
                    "sim": se.sim, "mjm": mjm, "mjd": mujoco.MjData(mjm),
                    "origins": se.scene.env_origins.cpu().numpy(), "free": fq,
                    "n": se.sim.data.qpos.shape[0],
                })

            # Round-robin worlds across sub-envs → a mix of tools, up to 16 tiles.
            self._display: list[tuple[int, int]] = []
            wptr = [0] * len(self._subs)
            while len(self._display) < 16 and any(
                wptr[s] < self._subs[s]["n"] for s in range(len(self._subs))
            ):
                for s in range(len(self._subs)):
                    if wptr[s] < self._subs[s]["n"] and len(self._display) < 16:
                        self._display.append((s, wptr[s]))
                        wptr[s] += 1
            k = len(self._display)

            spacing = 1.3
            cols = max(1, int(math.ceil(math.sqrt(k))))
            rows = int(math.ceil(k / cols))
            self._grid = np.zeros((k, 3), dtype=np.float64)
            for i in range(k):
                r, c = divmod(i, cols)
                self._grid[i, 0] = (c - (cols - 1) / 2.0) * spacing
                self._grid[i, 1] = (r - (rows - 1) / 2.0) * spacing

            self._renderer = mujoco.Renderer(
                self._subs[0]["mjm"], 600, 900, max_geom=40000
            )
            self._opt = mujoco.MjvOption()
            self._pert = mujoco.MjvPerturb()
            self._catmask = int(mujoco.mjtCatBit.mjCAT_DYNAMIC) | int(
                mujoco.mjtCatBit.mjCAT_STATIC
            )
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0.2, -0.12, 0.95]
            self._cam.distance = max(2.0, cols * spacing * 1.25)
            self._cam.azimuth = 120
            self._cam.elevation = -18
            self._win = f"SimToolReal-G1  ({k} envs, mixed tools, live)"
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, 900, 600)
            self._viewer = True

    def _set_world(self, sub: dict, i: int) -> None:
        import mujoco

        d = sub["mjd"]
        d.qpos[:] = sub["sim"].data.qpos[i].cpu().numpy()
        d.qvel[:] = sub["sim"].data.qvel[i].cpu().numpy()
        if sub["free"] is not None:
            d.qpos[sub["free"] : sub["free"] + 3] -= sub["origins"][i]
        mujoco.mj_forward(sub["mjm"], d)

    def _offset_geoms(self, start: int, slot: int) -> None:
        scene = self._renderer.scene
        for g in range(start, scene.ngeom):
            scene.geoms[g].pos[:] += self._grid[slot]

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            return
        self._view_ct += 1
        if self._view_ct % self._view_every:
            return
        import mujoco

        # First tile seeds the scene; the rest are appended (any sub-env model).
        s0, w0 = self._display[0]
        sub0 = self._subs[s0]
        self._set_world(sub0, w0)
        self._renderer.update_scene(sub0["mjd"], self._cam)
        self._offset_geoms(0, 0)
        for slot in range(1, len(self._display)):
            s, w = self._display[slot]
            sub = self._subs[s]
            self._set_world(sub, w)
            n0 = self._renderer.scene.ngeom
            mujoco.mjv_addGeoms(
                sub["mjm"], sub["mjd"], self._opt, self._pert,
                self._catmask, self._renderer.scene,
            )
            self._offset_geoms(n0, slot)
        rgb = self._renderer.render()
        self._cv2.imshow(self._win, rgb[:, :, ::-1])  # RGB -> BGR for OpenCV
        self._cv2.waitKey(1)

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self) -> SimToolRealG1Env:
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def _to_td(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        return TensorDict(
            cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]
        )

    def get_observations(self) -> TensorDict:
        return self._to_td(self.env.obs_buf)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return self._to_td(obs_dict), extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # Bootstrap value at time-limit truncations (not true terminals).
        extras["time_outs"] = truncated
        self._sync_viewer()
        return self._to_td(obs_dict), rew, dones, extras

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._renderer.close()
                self._cv2.destroyWindow(self._win)
            except Exception:
                pass
