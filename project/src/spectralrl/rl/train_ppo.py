"""PPO training loop on a Gymnasium env, implemented with TorchRL.

Consumes a YAML config like ``configs/ppo_default.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from tensordict.nn import TensorDictModule  # noqa: F401 (used indirectly)
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs import EnvBase, ObservationNorm, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from ..utils.logging import CSVLogger, save_manifest
from ..utils.seeding import set_seed
from .policy import build_actor, build_critic


@dataclass
class TrainResult:
    run_dir: Path
    best_ckpt: Path
    final_reward: float


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _wrap_env(env_factory: Callable[[], "gym.Env"], device: str) -> EnvBase:
    base = GymWrapper(env_factory(), device=device)
    env = TransformedEnv(base, StepCounter())
    obs_norm = ObservationNorm(in_keys=["observation"])
    env.append_transform(obs_norm)
    env.transform[1].init_stats(num_iter=512, reduce_dim=0, cat_dim=0)
    return env


def train(
    env_factory: Callable[[], "gym.Env"],
    config: dict | str | Path,
    out_dir: str | Path,
) -> TrainResult:
    if isinstance(config, (str, Path)):
        config = _load_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(out_dir, config)

    seed = int(config.get("seed", 0))
    device = str(config.get("device", "cpu"))
    set_seed(seed)
    ppo_cfg = config["ppo"]

    env = _wrap_env(env_factory, device)
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    act_dim = int(env.action_spec.shape[-1])
    hidden = tuple(ppo_cfg.get("hidden_sizes", [128, 128]))
    actor = build_actor(obs_dim, act_dim, hidden).to(device)
    critic = build_critic(obs_dim, hidden).to(device)

    # Prime the critic so GAE has shape info before the first rollout.
    with torch.no_grad():
        td0 = env.reset()
        critic(td0)

    advantage = GAE(
        gamma=float(ppo_cfg["gamma"]),
        lmbda=float(ppo_cfg["lmbda"]),
        value_network=critic,
        average_gae=True,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=float(ppo_cfg["clip_epsilon"]),
        entropy_coeff=float(ppo_cfg.get("entropy_coef", 0.0)),
        critic_coeff=float(ppo_cfg.get("critic_coef", 1.0)),
        loss_critic_type="smooth_l1",
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=float(ppo_cfg["lr"]))

    frames_per_batch = int(ppo_cfg["frames_per_batch"])
    sub_batch_size = int(ppo_cfg["sub_batch_size"])
    total_frames = int(ppo_cfg["total_frames"])
    num_epochs = int(ppo_cfg["num_epochs"])
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 1.0))

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=sub_batch_size,
    )

    logger = CSVLogger(
        out_dir / "train_log.csv",
        fieldnames=["iter", "frames", "mean_reward", "loss_total", "loss_policy", "loss_value"],
    )
    best = -float("inf")
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    it = 0
    for tensordict_data in collector:
        it += 1
        with torch.no_grad():
            advantage(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        rb.empty()
        rb.extend(data_view.cpu())
        losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
        updates = 0
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // sub_batch_size):
                sub = rb.sample().to(device)
                loss_td = loss_module(sub)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td.get("loss_entropy", torch.tensor(0.0, device=device))
                )
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                losses["total"] += float(loss.item())
                losses["policy"] += float(loss_td["loss_objective"].item())
                losses["value"] += float(loss_td["loss_critic"].item())
                updates += 1

        mean_r = float(tensordict_data["next", "reward"].mean().item())
        frames = it * frames_per_batch
        logger.log(
            {
                "iter": it,
                "frames": frames,
                "mean_reward": mean_r,
                "loss_total": losses["total"] / max(updates, 1),
                "loss_policy": losses["policy"] / max(updates, 1),
                "loss_value": losses["value"] / max(updates, 1),
            }
        )
        if mean_r > best:
            best = mean_r
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "config": config,
                    "obs_norm": env.transform[1].state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
                best_ckpt,
            )
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": config,
                "obs_norm": env.transform[1].state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            },
            last_ckpt,
        )

    return TrainResult(run_dir=out_dir, best_ckpt=best_ckpt, final_reward=best)


@torch.no_grad()
def rollout_deterministic(actor, env_factory: Callable[[], "gym.Env"], n_episodes: int = 16) -> dict[str, float]:
    """Deterministic eval rollout using the mean of the policy distribution."""
    rewards, lambda2s, costs = [], [], []
    for _ in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset()
        total = 0.0
        done = False
        truncated = False
        last_info: dict[str, float] = {}
        while not (done or truncated):
            with set_exploration_type(ExplorationType.MODE):
                import torch as _t

                from tensordict import TensorDict

                td = TensorDict(
                    {"observation": _t.tensor(np.asarray(obs), dtype=_t.float32).unsqueeze(0)},
                    batch_size=[1],
                )
                td = actor(td)
                action = td["action"].squeeze(0).cpu().numpy()
            obs, r, done, truncated, info = env.step(action)
            total += float(r)
            last_info = info
        rewards.append(total)
        lambda2s.append(float(last_info.get("lambda2", 0.0)))
        costs.append(float(last_info.get("cost", 0.0)))
    return {
        "mean_return": float(np.mean(rewards)),
        "mean_lambda2": float(np.mean(lambda2s)),
        "mean_cost": float(np.mean(costs)),
    }
