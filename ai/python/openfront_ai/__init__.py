"""OpenFront AI training package — APPO / BC against Nations via TS env."""

from .env import OpenFrontEnv, make_vec_env
from .policy import MacroMicroPolicy
from .rewards import shaped_reward_torch

__all__ = [
    "OpenFrontEnv",
    "make_vec_env",
    "MacroMicroPolicy",
    "shaped_reward_torch",
]
