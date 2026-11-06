# Isaac Lab 3.x migration

This external task package targets **Isaac Lab main**, **Isaac Sim 6.2**, the
**Isaac Sim PhysX backend**, and **RSL-RL PPO**. The changes have passed
configuration and packaging checks. **Isaac Sim physics, rendering, and PPO
training have not been executed or validated.** This cloud machine has no
NVIDIA GPU or Isaac Sim runtime.

## Audited versions and official sources

- Project baseline: `bcb4ec233927beb6e8b1806fad4703f52131dcbb`.
- Migration branch: `migrate/isaaclab-3x-physx`.
- Isaac Lab main audited at `b3ef9ce18ef640fe27fcd18c1527382897a94109`.
- That workspace pins Isaac Sim `6.2.0.0`, PyTorch `2.11.0`,
  Warp `1.17.0`, and RSL-RL `5.5.1`, using Python `>=3.12,<3.13`.
- Individual workspace packages have independent version numbers:
  `isaaclab` 44.0.0, `isaaclab_physx` 11.0.0, `isaaclab_rl` 5.3.1,
  `isaaclab_tasks` 30.1.1. These are not the Isaac Lab release number.
  The external package declares these as dependency floors; use the complete
  official workspace, including its dependency overrides, rather than resolving
  an arbitrary set of standalone packages.

The official [migration skill](https://github.com/isaac-sim/IsaacLab/blob/b3ef9ce18ef640fe27fcd18c1527382897a94109/skills/user/migrate-2x-to-3x/SKILL.md)
and its reference, supplemental checks, examples, and evaluations were read.
API decisions were checked against the official
[migration guide](https://github.com/isaac-sim/IsaacLab/blob/b3ef9ce18ef640fe27fcd18c1527382897a94109/docs/source/isaaclab/migrating.rst)
and implementations at that revision:

- `source/isaaclab/isaaclab/actuators/actuator_cfg.py`
- `source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py`
- `source/isaaclab_physx/isaaclab_physx/sim/schemas/schemas_cfg.py`
- `source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/{observations,rewards,events,terminations}.py`
- `source/isaaclab/isaaclab/utils/noise/noise_cfg.py`
- `source/isaaclab/isaaclab/visualizers/visualizer_cfg.py`
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
- `source/isaaclab_rl/isaaclab_rl/entrypoints/{dispatch,common}.py`
- `source/isaaclab_rl/isaaclab_rl/entrypoints/backends/{train,play}_rsl_rl.py`
- `source/isaaclab/isaaclab/cli/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/curriculums.py`
- Current Cartpole/ANYmal task examples and the external package template.

No framework files were changed. Future main revisions may require additional
changes; record the framework SHA when running the acceptance tests.

## API mappings

| 2.x usage | Current main usage | Behavior preserved |
| --- | --- | --- |
| `RigidBodyPropertiesCfg` | `rigid_props=[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]` | Rigid-body enable, velocity limits, depenetration and gyroscopic forces |
| `ArticulationRootPropertiesCfg` | `articulation_props=[PhysxArticulationCfg(...)]` | Self collisions disabled, 8/0 solver iterations, sleep/stabilization thresholds |
| `effort_limit_sim` | `joint_effort_limit` | Wheels 100, thighs 400 |
| `velocity_limit_sim` | `joint_velocity_limit` | All four joints 100 |
| Implicit actuator gains | `ImplicitActuatorCfg` with canonical solver-limit fields | Wheels stiffness/damping 0/10, thighs 1000/20 |
| Implicit backend default | `SimulationCfg.physics` with a `PresetCfg` exposing `isaacsim_physx` | Only Isaac Sim PhysX is offered and is the default |
| `AdditiveUniformNoiseCfg` | `UniformNoiseCfg` (`operation="add"` by default) | Original noise intervals |
| `RigidBodyMaterialCfg` for rough ground | `PhysxRigidBodyMaterialCfg` | Friction/restitution coefficients and combine modes |
| `self.viewer.eye` | `self.sim.visualizer_cfgs = VisualizerCfg(eye=...)` | Original viewing position; CLI controls whether a viewer opens |
| WXYZ config quaternions | XYZW | Light `(0.738, .477, .477, 0)` becomes `(.477, .477, 0, .738)`; robot identity explicitly `(0,0,0,1)` |
| `RslRlPpoActorCriticCfg` under `policy` | Separate `actor` and `critic` `RslRlMLPModelCfg` | ELU MLP widths and disabled observation normalization |
| `init_noise_std=1.0` | Actor `GaussianDistributionCfg(init_std=1.0, std_type="scalar")` | Initial action noise and scalar standard-deviation parameterization |
| `obs_groups={"policy": ["policy"], ...}` | `{"actor": ["policy"], "critic": ["critic"]}` | Onboard-only actor; privileged critic |
| `empirical_normalization=False` | Each model's `obs_normalization=False` | No observation normalization |
| `isaaclab_tasks.manager_based.locomotion.velocity.mdp` | `isaaclab_tasks.core.velocity.mdp` | Terrain distance/velocity curriculum; upstream uses `.torch` for ProxyArray data |
| Import root package for registration | Installed `isaaclab.tasks` entry point to `wheeled_quadruped.tasks` | External task discovery by standard CLI; root package is passive |
| Task IDs ending `-v0` | Suffixless task IDs | Same nine task/config pairs, including Play, rough, and legacy Custom alias |
| Multiple legacy RL registration keys | `rsl_rl_cfg_entry_point`, `default_agent="rsl_rl"` | Only migrated RSL-RL is advertised; legacy YAML files are retained |
| Copied train/play launchers | `uv run ... isaaclab train` / `play` | Maintained runner/wrapper handles current actuator and ProxyArray APIs |

The MDP uses upstream built-in terms, not custom tensor operations. Their
current implementations already access ProxyArray through `.torch` and use
the current actuator ownership APIs. No downstream data-access rewrite was
necessary. The custom legacy scripts still contain 2.x assumptions and are
explicitly outside the migrated workflow.

## Preserved task behavior and assets

- Original four-legged/wheeled robot, USD, URDF, meshes, and pretrained files.
  No biped conversion. The USD is byte-identical to the baseline; SHA-256:
  `205bd17dc2dd899c30a4e24ee4575fcd18d068b0139cfa5f563e19719b3f0e36`.
- OpenUSD inspection finds the same four revolute joints: front left/right
  thighs and rear left/right wheels. No new root rotation was introduced.
- Initial base height 0.828 m; physics timestep 0.005 s; decimation 4;
  20 s / 1000 control-step episodes.
- Action term order: thighs then wheels. Scales remain 0.5 for thigh position,
  5 for balance wheel velocity, and 12 for velocity-task wheel velocity.
- Original observation term order and expected dimensions: balance actor/critic
  16/20, velocity actor/critic 19/23. Actual runtime joint ordering and tensor
  dimensions still require checking in PhysX.
- Original rewards, termination criteria, randomizations, command ranges,
  rough terrain mix, curriculum, and Play settings are retained. Runtime parity
  is not established by preserving config values.
- PPO rollout 24, learning rate 0.001, four minibatches, five learning epochs,
  original clipping/discount/GAE/entropy and iteration budgets are retained.
  Changing the RSL-RL major version does not guarantee numerically identical training.
- The old actor/checkpoint state layout is not assumed compatible with RSL-RL 5.
  No checkpoint conversion was attempted. Use newly trained checkpoints for
  the 3.x play/resume tests.

## Modified files and reasons

Paths below are relative to the repository root. Package paths are under
`source/wheeled_quadruped/`.

| File | Reason |
| --- | --- |
| `README.md` | Current setup/train/play commands, suffixless task IDs, and clear legacy/result limitations |
| `source/wheeled_quadruped/pyproject.toml` | Python 3.12, version 0.4.0, framework dependencies and external discovery entry point |
| `wheeled_quadruped/__init__.py` | Passive package import; registration through task entry point |
| `wheeled_quadruped/assets/__init__.py` | Schema fragments, canonical joint limits, explicit XYZW identity |
| `wheeled_quadruped/tasks/balance/balance_env_cfg.py` | Noise rename, light quaternion, PhysX preset, visualizer configuration |
| `wheeled_quadruped/tasks/balance/rough_env_cfg.py` | Backend-owned physics material configuration |
| `wheeled_quadruped/tasks/velocity/rough_env_cfg.py` | Current terrain curriculum import path |
| `wheeled_quadruped/tasks/balance/agents/rsl_rl_ppo_cfg.py` | RSL-RL 5 actor/critic model and observation-set contracts |
| `wheeled_quadruped/tasks/velocity/agents/rsl_rl_ppo_cfg.py` | Same migration with original velocity network widths |
| `wheeled_quadruped/tasks/balance/__init__.py` | Suffixless task registrations and RSL-RL default |
| `wheeled_quadruped/tasks/velocity/__init__.py` | Suffixless task registrations and RSL-RL default |
| `tests/test_migration_config.py` | Framework-backed configuration/discovery/API signature checks |
| `docs/MIGRATION_ISAACLAB_3.md` | Migration evidence and outstanding acceptance tests |

The flat velocity MDP inherits the migrated balance settings; its original
configuration file requires no direct edits. The terrain definitions are unchanged.
Custom `scripts/rsl_rl/train.py`, `play.py`, `scripts/verify_env.py`, and other
legacy scripts were not modified.

## Verification performed

Verified in a separate Python 3.12 CPU environment against the unmodified
framework source revision above. This was a **partial, configuration-only
dependency environment**, not a full Isaac Lab / Isaac Sim installation.

| Check | Result |
| --- | --- |
| Real framework config tests | **29 passed**, zero failed/skipped; all nine task registrations, default/explicit PhysX selection, config validation, model/MDP invariants, and term signatures |
| Test warnings | 35 upstream PyTorch TorchScript deprecation warnings |
| AST parse | 15 project/test Python files parsed successfully (excluding generated build outputs) |
| Ruff fatal syntax/undefined-name checks | Passed: `ruff check --select E9,F63,F7,F82 source/wheeled_quadruped tests` |
| Official quaternion finder | Both intended source quaternion sites identified as changed; zero unchanged candidates; manually reviewed |
| Editable package build/install | Succeeded with `--no-deps` in the configuration-check environment |
| Wheel build | Succeeded; packaged USD bytes and external-task entry point verified; setuptools emitted a pre-existing license-table deprecation warning |
| USD structure/identity | Opened with OpenUSD; four expected joint prims; asset bytes unchanged |
| Standard CLI discovery | Real `isaaclab list_envs --keyword Wheeled-Quadruped --show_presets` implementation listed all nine tasks with RSL-RL and PhysX default (source-import CPU environment) |
| Diff hygiene | `git diff --check` passed |
| PhysX stepping, GPU, GUI, PPO | **Not run** |

The test runner made the official checkout's `source/*` packages importable
using Python's import path; no stubbed framework APIs were used. To reproduce
the tests in a complete internal container, use the commands below instead of
reconstructing that partial environment.

## Internal Isaac Lab container acceptance tests still required

Use the internal GPU container with Isaac Sim 6.2 and a fresh Isaac Lab main
workspace. Do not modify the framework or install into an old 2.x environment.
From that workspace (adjust the external repository mount):

```bash
export ROBOT_PACKAGE=/workspace/wheeled_quadruped_robot/source/wheeled_quadruped
uv sync --extra isaacsim
uv run --extra isaacsim --with-editable "$ROBOT_PACKAGE" isaaclab list_envs
uv run --extra isaacsim --with-editable "$ROBOT_PACKAGE" --with pytest python -m pytest /workspace/wheeled_quadruped_robot/tests/test_migration_config.py -q
```

1. **Discovery/config composition:** confirm all nine task IDs, the RSL-RL
   agent, PhysX default, and `physics=isaacsim_physx`. The `--with-editable`
   argument keeps the external distribution present even when uv syncs the
   workspace. Record framework SHA, GPU, driver, simulator and package versions.
2. **Asset initialization:** inspect the spawned articulation in PhysX. Verify
   four actuated joints, fixed front wheels, original wheel contacts, upright
   stance, 0.828 m initial root height, joint drive gains/limits, mass/material
   randomization, and solver iterations. The USD contents were not converted.
3. **Random-action stepping:** run each flat and rough task at 8 environments:

   ```bash
   for task in Wheeled-Quadruped-Balance Wheeled-Quadruped-Velocity Wheeled-Quadruped-Balance-Rough Wheeled-Quadruped-Velocity-Rough; do
     uv run --extra isaacsim --with-editable "$ROBOT_PACKAGE" isaaclab random_agent --task "$task" --num_envs 8 --max_steps 1000 physics=isaacsim_physx visualizer=none
   done
   ```

   Beyond exit status, inspect observations/rewards/actions for NaN/Inf,
   confirm action order `[FL_thigh, FR_thigh, rl_wheel, rr_wheel]`, joint-velocity
   observation ordering, policy/critic dimensions 16/20 or 19/23, automatic
   resets, startup randomizations, pushes after 10–15 s, the 0.4 m height and
   60-degree tilt terminations, and 1000-step timeouts. Random-action falls
   are expected; survival is not a smoke-test acceptance criterion.
4. **PPO updates:** execute at least two iterations for all four tasks at 32
   environments (32 × 24 is divisible by four minibatches):

   ```bash
   for task in Wheeled-Quadruped-Balance Wheeled-Quadruped-Velocity Wheeled-Quadruped-Balance-Rough Wheeled-Quadruped-Velocity-Rough; do
     uv run --extra isaacsim --with-editable "$ROBOT_PACKAGE" isaaclab train --rl_library rsl_rl --task "$task" --num_envs 32 --max_iterations 2 physics=isaacsim_physx visualizer=none
   done
   ```

   Confirm actual rollout collection and PPO updates, finite losses, actor and
   privileged critic input sizes, and saved config/checkpoint files. Then
   resume a newly generated checkpoint with `--checkpoint /absolute/path/model.pt`.
   Two iterations validate integration, not convergence or control quality.
5. **Play/GUI:** use a new 3.x checkpoint, not the historical shipped checkpoint:

   ```bash
   uv run --extra isaacsim --with-editable "$ROBOT_PACKAGE" isaaclab play --rl_library rsl_rl --task Wheeled-Quadruped-Velocity-Play --num_envs 8 --checkpoint /absolute/path/to/new/model.pt physics=isaacsim_physx visualizer=kit
   ```

   Verify the original light/camera orientation, noise/push disablement, fixed
   0.5 m/s forward command, and reset behavior. Repeat for Balance-Play and
   both Rough-Play variants. GUI requires a working container display.
6. **Behavior/parity:** train longer runs and compare survival, height/tilt,
   commanded-versus-achieved linear/yaw speed, wheel contacts, torque limits,
   and terrain curriculum progression. No historical score is an acceptance
   result for the new simulator. Inspect velocity-rough separately because the
   original policy already had lower survival.

The legacy verifier was intentionally not migrated. Use the maintained standard
CLI and container diagnostics for these runtime checks; the configuration tests
cannot substitute for them.
