from __future__ import annotations
import copy
import torch
from .env_cfg import EnvCfg
from .simtoolreal_env import SimToolRealG1Env

def compute_split(total: int, ratios: tuple[int, ...]) -> list[int]:
    k = len(ratios)
    if total < k:
        raise ValueError(f"num_envs={total} < number of topologies={k}")
    s = float(sum(ratios))
    counts = [max(1, int(total * r / s)) for r in ratios]
    order = sorted(range(k), key=lambda i: ratios[i], reverse=True)
    diff = total - sum(counts)
    i = 0
    while diff > 0:
        counts[order[i % k]] += 1
        diff -= 1
        i += 1
    while diff < 0:
        j = order[i % k]
        if counts[j] > 1:
            counts[j] -= 1
            diff += 1
        i += 1
    return counts

class MultiObjectEnv:
    def __init__(
        self,
        cfg: EnvCfg,
        device: str | None = None,
        topologies: tuple[str, ...] | None = None,
        split: tuple[int, ...] | None = None,
    ):
        self.cfg = cfg
        self._device = device or cfg.device
        self.extras: dict = {}

        topologies = topologies or cfg.object.topologies
        split = split or cfg.object.topology_split
        assert len(topologies) == len(split), "topologies/split length mismatch"

        counts = compute_split(cfg.scene.num_envs, tuple(split))
        self.topologies = tuple(topologies)
        self.sub_counts = counts

        self.sub_envs: list[SimToolRealG1Env] = []
        for topo, nk in zip(topologies, counts):
            sub_cfg = copy.deepcopy(cfg)
            sub_cfg.scene.num_envs = nk
            sub_cfg.device = self._device
            self.sub_envs.append(
                SimToolRealG1Env(sub_cfg, device=self._device, topology=topo)
            )

        # Contiguous global<->sub index blocks.
        bounds = [0]
        for nk in counts:
            bounds.append(bounds[-1] + nk)
        self._bounds = bounds  # len K+1; sub k -> [bounds[k], bounds[k+1])
        self._num_envs = bounds[-1]

        # Prime obs_buf for get_observations().
        self.obs_buf, _ = self.reset()

    # --- interface the vecenv wrapper reads ---
    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def physics_dt(self) -> float:
        return self.sub_envs[0].physics_dt

    @property
    def step_dt(self) -> float:
        return self.sub_envs[0].step_dt

    @property
    def max_episode_length(self) -> int:
        return self.sub_envs[0].max_episode_length

    @property
    def sim(self):
        # env 0 lives in sub-env 0; the live viewer follows sub-env 0's world 0.
        return self.sub_envs[0].sim

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat([se.episode_length_buf for se in self.sub_envs])

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        for k, se in enumerate(self.sub_envs):
            se.episode_length_buf = value[self._bounds[k]:self._bounds[k + 1]].clone()

    # --- helpers ---
    def _cat_obs(self, obs_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = obs_list[0].keys()
        return {key: torch.cat([o[key] for o in obs_list], dim=0) for key in keys}

    def _merge_extras(self, extras_list: list[dict]) -> dict:
        merged: dict = {}
        ep_keys = extras_list[0].get("episode", {}).keys()
        if ep_keys:
            merged["episode"] = {
                k: torch.cat([e["episode"][k] for e in extras_list], dim=0)
                for k in ep_keys
            }
        return merged

    # --- core loop ---
    def step(self, actions: torch.Tensor):
        obs_list, rew_list, term_list, trunc_list, extras_list = [], [], [], [], []
        for k, se in enumerate(self.sub_envs):
            a = actions[self._bounds[k]:self._bounds[k + 1]]
            obs_k, rew_k, term_k, trunc_k, extras_k = se.step(a)
            obs_list.append(obs_k)
            rew_list.append(rew_k)
            term_list.append(term_k)
            trunc_list.append(trunc_k)
            extras_list.append(extras_k)

        self.obs_buf = self._cat_obs(obs_list)
        reward = torch.cat(rew_list, dim=0)
        terminated = torch.cat(term_list, dim=0)
        truncated = torch.cat(trunc_list, dim=0)
        self.extras = self._merge_extras(extras_list)
        return self.obs_buf, reward, terminated, truncated, self.extras

    def reset(self):
        obs_list, extras_list = [], []
        for se in self.sub_envs:
            obs_k, extras_k = se.reset()
            obs_list.append(obs_k)
            extras_list.append(extras_k)
        self.obs_buf = self._cat_obs(obs_list)
        self.extras = self._merge_extras(extras_list)
        return self.obs_buf, self.extras
