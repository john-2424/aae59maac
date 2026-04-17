"""Actor/critic modules for PPO on continuous edge reweighting.

Uses TorchRL's ``ProbabilisticActor`` with a ``TanhNormal`` so that actions
stay in ``[-1, 1]`` — the env scales them to ``[0, w_max]`` internally.
"""

from __future__ import annotations

from typing import Sequence

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


def build_actor(obs_dim: int, act_dim: int, hidden: Sequence[int] = (128, 128)) -> ProbabilisticActor:
    backbone = _mlp(obs_dim, 2 * act_dim, hidden)
    head = nn.Sequential(backbone, NormalParamExtractor())
    base = TensorDictModule(head, in_keys=["observation"], out_keys=["loc", "scale"])
    return ProbabilisticActor(
        module=base,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )


def build_critic(obs_dim: int, hidden: Sequence[int] = (128, 128)) -> ValueOperator:
    net = _mlp(obs_dim, 1, hidden)
    return ValueOperator(module=net, in_keys=["observation"])


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad()
def deterministic_action(actor: ProbabilisticActor, obs: torch.Tensor) -> torch.Tensor:
    """Return ``tanh(loc)`` — a mode-like deterministic action for evaluation."""
    td = actor.module(_wrap_obs(obs))
    loc = td["loc"]
    return torch.tanh(loc)


def _wrap_obs(obs: torch.Tensor):
    from tensordict import TensorDict

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    return TensorDict({"observation": obs}, batch_size=obs.shape[:1])
