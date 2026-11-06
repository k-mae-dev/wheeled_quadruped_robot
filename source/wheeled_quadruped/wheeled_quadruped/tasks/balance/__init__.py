# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheeled quadruped balancing environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Wheeled-Quadruped-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:WheeledQuadrupedBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledQuadrupedBalancePPORunnerCfg",
    },
)

# Historical Custom task name, now suffixless, mapped to the same balance configuration.
gym.register(
    id="Custom-Wheeled-Quadruped",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:WheeledQuadrupedBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledQuadrupedBalancePPORunnerCfg",
    },
)

gym.register(
    id="Wheeled-Quadruped-Balance-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:WheeledQuadrupedBalanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledQuadrupedBalancePPORunnerCfg",
    },
)

# --- Rough-terrain variants (uneven ground) ---
gym.register(
    id="Wheeled-Quadruped-Balance-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WheeledQuadrupedBalanceRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledQuadrupedBalanceRoughPPORunnerCfg",
    },
)

gym.register(
    id="Wheeled-Quadruped-Balance-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WheeledQuadrupedBalanceRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledQuadrupedBalanceRoughPPORunnerCfg",
    },
)
