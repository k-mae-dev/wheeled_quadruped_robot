"""Configuration-only migration checks; no simulation, physics, or PPO is run.

Run with the external package installed in the Isaac Lab main uv environment.
These tests deliberately use real framework classes, without API stubs.
"""

import importlib.metadata
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import gymnasium as gym
import pytest

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_tasks.utils import resolve_task_config

TASKS = [
    "Wheeled-Quadruped-Balance",
    "Custom-Wheeled-Quadruped",
    "Wheeled-Quadruped-Balance-Play",
    "Wheeled-Quadruped-Balance-Rough",
    "Wheeled-Quadruped-Balance-Rough-Play",
    "Wheeled-Quadruped-Velocity",
    "Wheeled-Quadruped-Velocity-Play",
    "Wheeled-Quadruped-Velocity-Rough",
    "Wheeled-Quadruped-Velocity-Rough-Play",
]


@pytest.fixture(scope="module", autouse=True)
def discover_external_tasks():
    entries = [
        entry for entry in importlib.metadata.entry_points(group="isaaclab.tasks") if entry.name == "wheeled_quadruped"
    ]
    assert len(entries) == 1, "Install source/wheeled_quadruped into the active environment first"
    assert entries[0].value == "wheeled_quadruped.tasks"
    entries[0].load()


def resolve(task, overrides=()):
    return resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=list(overrides))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("overrides", [(), ("physics=isaacsim_physx",)])
def test_registered_configs_resolve_and_validate(task, overrides):
    env, agent = resolve(task, overrides)
    assert gym.spec(task).kwargs["default_agent"] == "rsl_rl"
    assert isinstance(env.sim.physics, PhysxCfg)
    assert env.validate() == []
    assert agent.validate() == []
    assert env.sim.dt == 0.005
    assert env.decimation == env.sim.render_interval == 4
    assert env.episode_length_s == 20.0
    assert env.episode_length_s / (env.decimation * env.sim.dt) == 1000
    assert env.scene.num_envs == (32 if "Play" in task else 4096)
    assert env.scene.robot.init_state.pos == (0.0, 0.0, 0.828)
    assert env.scene.robot.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert env.scene.distant_light.init_state.rot == (0.477, 0.477, 0.0, 0.738)

    velocity = "Velocity" in task
    rough = "Rough" in task
    assert list(env.actions.__dict__) == ["thigh_pos", "wheel_vel"]
    assert env.actions.thigh_pos.scale == 0.5
    assert env.actions.wheel_vel.scale == (12.0 if velocity else 5.0)
    policy_terms = ["base_ang_vel", "projected_gravity", "thigh_pos", "joint_vel", "actions"]
    critic_terms = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "base_height",
        "thigh_pos",
        "joint_vel",
        "actions",
    ]
    if velocity:
        policy_terms.append("velocity_commands")
        critic_terms.append("velocity_commands")
        assert env.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
        assert env.rewards.wheel_spin is None
        assert env.rewards.track_ang_vel_z.weight == (0.5 if rough else 1.0)
    for group, expected in [(env.observations.policy, policy_terms), (env.observations.critic, critic_terms)]:
        assert [
            field.name for field in fields(group) if isinstance(getattr(group, field.name), ObservationTermCfg)
        ] == expected
        assert group.concatenate_terms
    assert not env.observations.critic.enable_corruption
    assert env.observations.policy.enable_corruption == ("Play" not in task)
    assert env.observations.policy.base_ang_vel.noise.operation == "add"
    assert env.rewards.base_height.params["target_height"] == 0.828
    assert env.rewards.base_height.weight == (-10.0 if velocity else -20.0)
    assert env.rewards.alive.weight == (0.25 if velocity else 1.0)
    assert env.rewards.flat_orientation.weight == (-2.0 if velocity else -5.0)
    assert env.terminations.base_too_low.params["minimum_height"] == 0.4
    assert env.terminations.time_out.time_out
    assert (env.events.push_robot is None) == ("Play" in task)
    if rough:
        assert env.scene.ground is None
        assert env.scene.terrain.terrain_generator.curriculum == (velocity and "Play" not in task)
    if velocity and "Play" in task:
        assert env.commands.base_velocity.ranges.lin_vel_x == (0.5, 0.5)
        assert env.commands.base_velocity.ranges.ang_vel_z == (0.0, 0.0)

    assert agent.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    dims = [256, 128, 64] if velocity else [128, 128]
    for model in (agent.actor, agent.critic):
        assert isinstance(model, RslRlMLPModelCfg)
        assert model.hidden_dims == dims
        assert model.activation == "elu"
        assert not model.obs_normalization
    assert agent.actor.distribution_cfg.init_std == 1.0
    assert agent.actor.distribution_cfg.std_type == "scalar"
    assert agent.critic.distribution_cfg is None
    assert agent.num_steps_per_env == 24
    assert agent.algorithm.class_name == "PPO"
    assert agent.algorithm.num_mini_batches == 4
    assert agent.algorithm.learning_rate == 1.0e-3
    assert agent.max_iterations == (3000 if velocity else 1500 if rough else 1000)


def test_articulation_schema_and_implicit_actuators():
    env, _ = resolve(TASKS[0])
    robot = env.scene.robot
    assert Path(robot.spawn.usd_path).is_file()
    assert isinstance(robot.spawn.rigid_props[0], UsdPhysicsRigidBodyCfg)
    assert isinstance(robot.spawn.rigid_props[1], PhysxRigidBodyCfg)
    assert isinstance(robot.spawn.articulation_props[0], PhysxArticulationCfg)
    assert robot.spawn.articulation_props[0].solver_position_iteration_count == 8
    assert not robot.spawn.articulation_props[0].enabled_self_collisions
    assert len(robot.actuators) == 4
    for name, actuator in robot.actuators.items():
        assert isinstance(actuator, ImplicitActuatorCfg)
        wheel = "wheel" in name
        assert actuator.joint_effort_limit == (100.0 if wheel else 400.0)
        assert actuator.joint_velocity_limit == 100.0
        assert actuator.stiffness == (0.0 if wheel else 1000.0)
        assert actuator.damping == (10.0 if wheel else 20.0)


@pytest.mark.parametrize("task", TASKS)
def test_mdp_term_parameters_match_current_function_signatures(task):
    """Catch renamed/removed parameters without constructing an environment."""
    env, _ = resolve(task)
    for group in (env.rewards, env.terminations):
        for field in fields(group):
            term = getattr(group, field.name)
            if isinstance(term, (RewardTermCfg, TerminationTermCfg)):
                inspect.signature(term.func).bind(None, **term.params)
    for group in (env.observations.policy, env.observations.critic):
        for field in fields(group):
            term = getattr(group, field.name)
            if isinstance(term, ObservationTermCfg):
                inspect.signature(term.func).bind(None, **term.params)
    for field in fields(env.events):
        term = getattr(env.events, field.name)
        if term is None:
            continue
        func = term.func.__call__ if inspect.isclass(term.func) else term.func
        args = (None, None, None) if inspect.isclass(term.func) else (None, None)
        inspect.signature(func).bind(*args, **term.params)
    if env.curriculum is not None and is_dataclass(env.curriculum):
        for field in fields(env.curriculum):
            term = getattr(env.curriculum, field.name)
            inspect.signature(term.func).bind(None, None, **term.params)


def test_config_instances_do_not_share_mutable_task_state():
    velocity, _ = resolve("Wheeled-Quadruped-Velocity")
    balance, _ = resolve("Wheeled-Quadruped-Balance")
    assert velocity.actions.wheel_vel.scale == 12.0
    assert balance.actions.wheel_vel.scale == 5.0
    assert balance.rewards.wheel_spin is not None
    rough, _ = resolve("Wheeled-Quadruped-Velocity-Rough")
    assert rough.rewards.track_ang_vel_z.weight == 0.5
    assert velocity.rewards.track_ang_vel_z.weight == 1.0
