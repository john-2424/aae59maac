"""Env package — submodules are imported lazily to avoid pulling in
``gymnasium`` / ``torchrl`` unless needed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ReweightEnv", "ReweightEnvConfig",
    "RewireEnv", "RewireEnvConfig",
    "GeometricSwarmEnv", "GeometricEnvConfig",
]

_MODULE_MAP = {
    "ReweightEnv": "reweight_env",
    "ReweightEnvConfig": "reweight_env",
    "RewireEnv": "rewire_env",
    "RewireEnvConfig": "rewire_env",
    "GeometricSwarmEnv": "geometric_env",
    "GeometricEnvConfig": "geometric_env",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        mod = import_module(f".{_MODULE_MAP[name]}", package=__name__)
        return getattr(mod, name)
    raise AttributeError(name)


if TYPE_CHECKING:
    from .geometric_env import GeometricEnvConfig, GeometricSwarmEnv
    from .rewire_env import RewireEnv, RewireEnvConfig
    from .reweight_env import ReweightEnv, ReweightEnvConfig
