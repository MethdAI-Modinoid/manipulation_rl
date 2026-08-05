# Dexterous Cube Grasping for the Unitree G1 with a BrainCo Revo2 Hand

Reinforcement learning for single-arm cube grasping on a Unitree G1 humanoid fitted
with a BrainCo Revo2 five-finger hand, trained in MuJoCo-Warp with SAPG.

The policy controls the right arm (7 DoF) and right hand (6 actuated DoF) of a
fixed-base G1, learning to reach, grasp, lift, and transport a cube from randomized
initial configurations under an automatic difficulty curriculum.

---

## Repository structure

```
.
├── scripts/          Entry points (setup, build, train, play, view)
├── rl/               RL package `simtoolreal_g1` — environment, reward, training
├── robot/            Robot model: MJCF, meshes, generation scripts
├── mjlab/            Simulation and RL framework (MuJoCo-Warp)
└── rl_games/         SAPG implementation
```

## Requirements

- Linux, NVIDIA GPU (developed on RTX 5090, CUDA 12.8)
- [`uv`](https://docs.astral.sh/uv/)

## Installation

```bash
git clone <repository-url> g1_grasp_repo
cd g1_grasp_repo
./scripts/setup.sh
```

`setup.sh` creates a pinned environment from `mjlab/uv.lock` and installs `mjlab`,
`rl_games`, and `simtoolreal_g1` as editable packages. No manual virtualenv or
`pip install` is required; do not activate an environment before running the scripts.

## Usage

**Train**

```bash
./scripts/train.sh --num-envs 3072 --num-blocks 6 --max-iterations 20000 --run-name RUN
```

Checkpoints are written to `mjlab/logs/rlg/<run>/nn/`.

**Evaluate**

```bash
./scripts/play.sh --checkpoint logs/rlg/<run>/nn/<ckpt>.pth \
    --num-envs 1 --steps 2000 --render --alpha 1.0
```

`--alpha` selects task difficulty and must match the value the checkpoint was trained
at. Omit `--render` for headless metrics.

**Regenerate the robot model** (only after morphology changes)

```bash
./scripts/build_model.sh
```

**Inspect the scene** (no policy)

```bash
./scripts/view.sh view_scene
./scripts/view.sh view_live --topology cyl_head
./scripts/view.sh verify_pose --alpha 0
```

## Task specification

| Property | Value |
|---|---|
| Action space | 13 continuous (7 arm joints, 6 finger joints), position targets |
| Observation space | 92-d |
| Critic state | Observations plus palm/object velocities, contact distances, lift flag, progress |
| Control frequency | 50 Hz policy, 200 Hz physics (`decimation = 4`) |
| Episode length | 1200 steps |
| Termination | Time limit; success tolerance follows a curriculum |

Observations comprise joint positions and velocities, previous action targets, palm
pose, object orientation, fingertip positions relative to the palm, and keypoints
relative to the palm and to the goal, clipped to ±10.

**Robot model.** `build_rl_model.py` derives the training MJCF from the full G1 +
Revo2 model by welding the floating base and all joints outside the right arm and
hand (34 welded, 18 retained), adding distal-follows-proximal finger couplings,
restricting collision to the hand, cube, and table, and enabling gravity compensation
on the right-arm chain. The 5 distal finger joints are passive.

## Reward

| Term | Weight | Description |
|---|---|---|
| `distance_delta` | 30.0 | Ratcheted fingertip-to-object progress |
| `bring_to_goal` | 20.0 | Gaussian object-to-goal distance, gated on lift |
| `lifting_rew` | 20.0 | Grasp-gated lift |
| `lifting_bonus` | 5.0 | Bonus above a 5 cm lift threshold |
| `reach_object` | 1.0 | Palm-to-object guidance |
| `joint_pos_limit_penalty` | 1.0 | Joint limit violation |
| `shove_penalty` | 0.5 | Object displacement without grasp |
| `arm_actions_penalty` | 0.005 | Arm action magnitude |
| `hand_actions_penalty` | 0.003 | Hand action magnitude |

Several additional terms are implemented but disabled by default in `RewardCfg`;
each is annotated in `task/env_cfg.py` with the failure mode that motivated disabling
it.

## Curriculum

A scalar `alpha` anneals task difficulty as `alpha = min(1, t / T)` with
`T = 150000` policy steps, jointly scaling three quantities:

| `alpha` | Initial condition |
|---|---|
| 0 | Hand pre-positioned over a cradled cube (grasping only) |
| 1 | Neutral hover pose 16 cm above, randomized cube position and yaw (full reach-and-grasp) |

Evaluation must specify the matching `--alpha`; `ResetCfg.eval_pose_alpha` freezes the
curriculum at a fixed value when set.

## Algorithm

SAPG (split-and-aggregate policy gradient), implemented as PPO with off-policy
experience sharing across `num_blocks` environment blocks. Actor-critic with an LSTM
(1024 units) followed by an MLP of [1024, 1024, 512, 512] and ELU activations, using
an asymmetric central value function. Hyperparameters are in
`rl/simtoolreal_g1/rlg/simtoolreal_sapg.yaml`; run-dependent fields (`num_actors`,
`minibatch_size`, `max_epochs`, `expl_coef_block_size`) are set from command-line
arguments at launch.

## Configuration

| File | Contents |
|---|---|
| `rl/simtoolreal_g1/task/env_cfg.py` | Scene, object, observations, reward, resets, terminations, domain randomization |
| `rl/simtoolreal_g1/rlg/simtoolreal_sapg.yaml` | SAPG/PPO hyperparameters |
| `rl/simtoolreal_g1/robot_cfg.py` | Actuator groups, gains, initial pose |

## Notes

- Environments must divide evenly by `--num-blocks`.
- `play.sh` restores the training-time environment configuration from `env_cfg.pkl` in
  the run directory; a warning is emitted if it is absent.
- Rendering during concurrent training on the same GPU may cause a GL/CUDA context
  conflict; use `--device cpu` or record with `--video`.
- Missing dependencies must be added to `rl/pyproject.toml` and re-locked, not
  installed with `uv pip install`, which the next sync will remove.

## Acknowledgements

Built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp RL framework) and
[rl_games](https://github.com/Denys88/rl_games) (SAPG fork). See `mjlab/LICENSE` and
`rl_games/LICENSE`.
