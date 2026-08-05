from __future__ import annotations

import gym
import numpy as np
import torch
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.ivecenv import IVecEnv

from ..task.env_cfg import EnvCfg
from ..task.multi_object_env import MultiObjectEnv
from ..task.simtoolreal_env import SimToolRealG1Env

import pickle
from pathlib import Path

def save_env_cfg(env_cfg, run_dir) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "env_cfg.pkl", "wb") as f:
        pickle.dump(env_cfg, f)

def load_env_cfg(run_dir):
    """Load a run's snapshotted EnvCfg (or None if absent — old runs)."""
    p = Path(run_dir) / "env_cfg.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def build_base_env(
    num_envs: int,
    device: str,
    multi_tool: bool,
    tool_split: str = "5,6,1",
    env_cfg=None,
):
    env_cfg = env_cfg if env_cfg is not None else EnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.device = device
    env_cfg.reward.reward_shaper_scale = 1.0  # rl_games applies the 0.01, not us
    if multi_tool:
        split = tuple(int(x) for x in tool_split.split(","))
        return MultiObjectEnv(env_cfg, device=device, split=split)
    topo = "box_only" if env_cfg.object.bootstrap_grasp else "box_head"
    return SimToolRealG1Env(env_cfg, device=device, topology=topo)

class SimToolRealRlGamesVecEnv(IVecEnv):

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        multi_tool: bool = False,
        tool_split: str = "5,6,1",
        clip_actions: float = 1.0,
        clip_observations: float = 10.0,
        env_cfg=None,
    ):
        self.env = build_base_env(num_envs, device, multi_tool, tool_split, env_cfg=env_cfg)
        self.cfg = self.env.cfg  # resolved EnvCfg (for train-time snapshotting)
        self.device = torch.device(device)
        self.num_envs = int(self.env.num_envs)
        self.num_actions = int(self.env.cfg.action.num_actions)
        self.clip_actions = clip_actions
        self.clip_observations = clip_observations

        obs_dict, _ = self.env.reset()
        self.obs_dim = int(obs_dict["policy"].shape[1])
        self.state_dim = int(obs_dict["critic"].shape[1])

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        inf = np.inf
        return {
            "observation_space": gym.spaces.Box(-inf, inf, (self.obs_dim,), dtype=np.float32),
            "action_space": gym.spaces.Box(-1.0, 1.0, (self.num_actions,), dtype=np.float32),
            "state_space": gym.spaces.Box(-inf, inf, (self.state_dim,), dtype=np.float32),
            "agents": 1,
            "value_size": 1,
        }

    def _obs(self, obs_dict: dict) -> dict:
        obs = obs_dict["policy"]
        states = obs_dict["critic"]
        if self.clip_observations is not None:
            c = self.clip_observations
            obs = torch.clamp(obs, -c, c)
            states = torch.clamp(states, -c, c)
        return {"obs": obs, "states": states}

    def reset(self) -> dict:
        obs_dict, _ = self.env.reset()
        return self._obs(obs_dict)

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(torch.uint8)
        infos = {"time_outs": truncated}  # rl_games value_bootstrap reads this
        if isinstance(extras, dict) and "episode" in extras:
            infos["episode"] = extras["episode"]
        return self._obs(obs_dict), rew, dones, infos


class EnvStatsAlgoObserver(AlgoObserver):
    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None
        self._sums: dict[str, float] = {}
        self._count = 0
        self._vmean = None
        self._rmean = None

    def after_init(self, algo):
        self.algo = algo
        self.writer = getattr(algo, "writer", None)
        _orig_prepare = algo.prepare_dataset

        def _wrapped(batch_dict):
            try:
                v = batch_dict.get("values")
                r = batch_dict.get("returns")
                if v is not None:
                    self._vmean = float(v.float().mean().item())
                if r is not None:
                    self._rmean = float(r.float().mean().item())
            except Exception:
                pass
            return _orig_prepare(batch_dict)

        algo.prepare_dataset = _wrapped

    def process_infos(self, infos, done_indices, **kwargs):
        if not isinstance(infos, dict):
            return
        ep = infos.get("episode")
        if not isinstance(ep, dict):
            return
        for k, v in ep.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    continue
                val = float(v.float().mean().item())
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
            self._sums[k] = self._sums.get(k, 0.0) + val
        self._count += 1

    def after_clear_stats(self):
        self._sums = {}
        self._count = 0

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return
        if self._vmean is not None:
            self.writer.add_scalar("value_fn/value_mean", self._vmean, frame)
        if self._rmean is not None:
            self.writer.add_scalar("value_fn/return_mean", self._rmean, frame)
        if self._count == 0:
            return
        for k, s in self._sums.items():
            self.writer.add_scalar(k, s / self._count, frame)
        self._sums = {}
        self._count = 0

__all__ = [
    "SimToolRealRlGamesVecEnv",
    "EnvStatsAlgoObserver",
    "build_base_env",
]
