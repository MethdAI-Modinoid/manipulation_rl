from __future__ import annotations

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg, Simulation

from .objects.object_gen import get_object_cfg
from ..robot_cfg import get_robot_cfg
from .env_cfg import EnvCfg
from .mdp import actions as _act
from .mdp import observations as _obs
from .mdp import rewards as _rew
from .mdp import reset_utils as _reset
from .mdp import terminations as _term

def _get_table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")  # no joint -> fixed base
    g = body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.15, 0.15, 0.02],
        rgba=[0.55, 0.4, 0.25, 1.0],
    )
    g.condim = 3  # default contact stiffness (stable for the light cube)
    return spec

def get_table_cfg(pos) -> EntityCfg:
    return EntityCfg(spec_fn=_get_table_spec,
                     init_state=EntityCfg.InitialStateCfg(pos=tuple(pos)))


def _bootstrap_cube_mass(obj_cfg) -> float:
    """Cube mass from the bootstrap size + density (vol * density)."""
    fx, fy, fz = obj_cfg.bootstrap_handle_full
    return float(fx * fy * fz * obj_cfg.bootstrap_density)      

def make_scene_cfg(cfg: EnvCfg, topology: str = "box_head") -> SceneCfg:
    return SceneCfg(
        num_envs=cfg.scene.num_envs,
        env_spacing=cfg.scene.env_spacing,
        entities={
            "robot": get_robot_cfg(),
            "object": get_object_cfg(
                topology, cfg.reset.object_start_pos,
                bootstrap_cube=(
                    (0.5 * cfg.object.bootstrap_handle_full[0],
                     _bootstrap_cube_mass(cfg.object))
                    if getattr(cfg.object, "bootstrap_grasp", False) else None
                ),
            ),
            "table": get_table_cfg(cfg.reset.table_pos),
        },
    )

class SimToolRealG1Env:
    def __init__(self, cfg: EnvCfg, device: str | None = None, topology: str = "box_head"):
        self.cfg = cfg
        self._device = device or cfg.device
        self._topology = topology
        self.extras: dict = {}

        self.scene = Scene(make_scene_cfg(cfg, topology), device=self._device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=SimulationCfg(

                mujoco=MujocoCfg(
                    timestep=cfg.sim_dt,
                    cone="elliptic",
                    impratio=10.0,
                    iterations=30,
                    ls_iterations=20,
                ),
                njmax=cfg.njmax,
                nconmax=cfg.nconmax,
            ),
            model=self.scene.compile(),
            device=self._device,
        )

        self.sim.expand_model_fields(_reset.OBJECT_DR_FIELDS)
        self.sim.create_graph()
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        _reset.sample_object_sizes(self, self._topology)
        _reset.allocate_state_buffers(self)

        if not getattr(self.cfg.object, "bootstrap_grasp", False):
            _reset.write_object_geometry(self, self._topology)
        self.reset()

    # Properties the MDP code reads.
    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def physics_dt(self) -> float:
        return self.cfg.sim_dt

    @property
    def step_dt(self) -> float:
        return self.cfg.sim_dt * self.cfg.decimation

    @property
    def max_episode_length(self) -> int:
        return self.cfg.termination.episode_length

    # Core loop.
    def step(self, actions: torch.Tensor):
        _act.apply_action_pipeline(self, actions)
        robot = self.scene["robot"]
        for _ in range(self.cfg.decimation):
            robot.set_joint_position_target(self._cur_targets, joint_ids=self._rl_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        rd = self.scene["robot"].data
        od = self.scene["object"].data
        _VEL_LIMIT = 200.0  # rad/s or m/s — far above any legitimate in-hand motion
        finite = (
            torch.isfinite(rd.joint_pos).all(-1)
            & torch.isfinite(rd.joint_vel).all(-1)
            & torch.isfinite(od.root_link_pos_w).all(-1)
            & torch.isfinite(od.root_link_lin_vel_w).all(-1)
            & torch.isfinite(od.root_link_ang_vel_w).all(-1)
        )
        tame = (
            (rd.joint_vel.abs().amax(-1) < _VEL_LIMIT)
            & (od.root_link_lin_vel_w.abs().amax(-1) < _VEL_LIMIT)
            & (od.root_link_ang_vel_w.abs().amax(-1) < _VEL_LIMIT)
        )
        diverged = ~(finite & tame)
        # ------------------------------------------------------------------------

        _obs.compute_intermediate_values(self)
        _reset.update_object_init_z(self)  # snapshot settled rest height as lift ref
        self.reward_buf = _rew.compute_rewards(self)
        terminated, truncated = _term.compute_terminations(self)
        _term.update_tolerance_curriculum(self)

        terminated = terminated | diverged
        self.reward_buf = torch.nan_to_num(
            self.reward_buf, nan=0.0, posinf=0.0, neginf=0.0
        )

        _skip = {"total_reward"}
        if self.cfg.reward.grasp_rew_scale == 0.0:
            _skip.add("grasp_rew")
        if self.cfg.reward.grasp_contact_scale == 0.0:
            _skip.add("grasp_contact_rew")
        if self.cfg.reward.reach_rew_scale == 0.0:
            _skip.add("reach_rew")
        self.extras["episode"] = {
            f"rew/{k}": torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in self._reward_terms.items()
            if k not in _skip
        }
        # A couple of useful task metrics alongside the rewards.
        self.extras["episode"]["stat/lifted_frac"] = self._lifted_object.float()
        self.extras["episode"]["stat/success_tolerance"] = torch.full_like(
            self.reward_buf, float(self._current_success_tolerance)
        )

        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.obs_buf = _obs.build_observations(self)
        self.obs_buf = {
            k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in self.obs_buf.items()
        }
        return self.obs_buf, self.reward_buf, terminated, truncated, self.extras

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        _reset.reset_env_state(self, env_ids)
        self.episode_length_buf[env_ids] = 0

    def reset(self):
        ids = torch.arange(self.num_envs, device=self._device)
        self._reset_idx(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        _obs.compute_intermediate_values(self)
        self.obs_buf = _obs.build_observations(self)
        return self.obs_buf, self.extras
