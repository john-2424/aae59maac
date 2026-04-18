"""Milestone 3a: train a discrete rewiring policy.

PPO with a categorical action head over all node pairs.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from spectralrl.envs.common import RewardConfig
from spectralrl.envs.rewire_env import RewireEnv, RewireEnvConfig
from spectralrl.graphs import erdos_renyi
from spectralrl.utils import set_seed


def make_env(config: dict):
    env_cfg = config["env"]
    n = int(env_cfg["n"])
    seed = int(config.get("seed", 0))
    p = float(env_cfg.get("er_p", 0.2))
    A, _ = erdos_renyi(n, p, seed=seed)
    # Budget sized off expected edge count when resampling, so it stays sane.
    resample = bool(env_cfg.get("resample_init", False))
    expected_edges = (n * (n - 1) / 2) * p
    base_edges = expected_edges if resample else float(np.triu(A > 0, k=1).sum())
    cfg = RewireEnvConfig(
        n=n,
        init_adj=A,
        edge_budget=int(1.5 * base_edges),
        degree_cap=None,
        episode_len=int(env_cfg.get("episode_len", 64)),
        reward=RewardConfig(
            beta=float(env_cfg["reward_beta"]), gamma=float(env_cfg["reward_gamma"])
        ),
        seed=seed,
        resample_init=resample,
        p_resample=p,
    )
    return RewireEnv(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/m3a"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--total-frames", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.n is not None:
        config["env"]["n"] = int(args.n)
    if args.total_frames is not None:
        config["ppo"]["total_frames"] = int(args.total_frames)

    set_seed(int(config.get("seed", 0)))

    # Build a discrete-action PPO trainer manually because the shared
    # ``train_ppo.train`` assumes a TanhNormal continuous actor. We wire a
    # categorical actor here with the same TorchRL plumbing.
    import torch
    from tensordict.nn import TensorDictModule
    from torch import nn
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import (
        LazyTensorStorage,
        SamplerWithoutReplacement,
        TensorDictReplayBuffer,
    )
    from torchrl.envs import StepCounter, TransformedEnv
    from torchrl.envs.libs.gym import GymWrapper
    from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE

    from spectralrl.utils.logging import CSVLogger, save_manifest

    save_manifest(args.out, config)
    device = "cpu"
    ppo = config["ppo"]

    base = GymWrapper(make_env(config), device=device)
    env = TransformedEnv(base, StepCounter())
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    n_actions = int(env.action_spec.space.n)
    hidden = list(ppo.get("hidden_sizes", [128, 128]))

    actor_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=hidden, activation_class=nn.Tanh)
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    critic_net = MLP(in_features=obs_dim, out_features=1, num_cells=hidden, activation_class=nn.Tanh)
    critic = ValueOperator(module=critic_net, in_keys=["observation"])

    advantage = GAE(
        gamma=float(ppo["gamma"]),
        lmbda=float(ppo["lmbda"]),
        value_network=critic,
        average_gae=True,
    )
    loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=float(ppo["clip_epsilon"]),
        entropy_coeff=float(ppo.get("entropy_coef", 0.01)),
        critic_coeff=float(ppo.get("critic_coef", 1.0)),
    )
    optim = torch.optim.Adam(loss.parameters(), lr=float(ppo["lr"]))

    frames_per_batch = int(ppo["frames_per_batch"])
    total_frames = int(ppo["total_frames"])
    sub_batch = int(ppo["sub_batch_size"])
    collector = SyncDataCollector(env, actor, frames_per_batch=frames_per_batch, total_frames=total_frames)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=sub_batch,
    )
    logger = CSVLogger(
        args.out / "train_log.csv",
        fieldnames=["iter", "mean_reward", "loss_total"],
    )

    it = 0
    best = -float("inf")
    for td_data in collector:
        it += 1
        with torch.no_grad():
            advantage(td_data)
        flat = td_data.reshape(-1)
        rb.empty()
        rb.extend(flat.cpu())
        total_loss = 0.0
        updates = 0
        for _ in range(int(ppo["num_epochs"])):
            for _ in range(frames_per_batch // sub_batch):
                sub = rb.sample().to(device)
                out = loss(sub)
                L = out["loss_objective"] + out["loss_critic"] + out.get(
                    "loss_entropy", torch.tensor(0.0)
                )
                optim.zero_grad()
                L.backward()
                nn.utils.clip_grad_norm_(loss.parameters(), float(ppo.get("max_grad_norm", 1.0)))
                optim.step()
                total_loss += float(L.item())
                updates += 1
        mean_r = float(td_data["next", "reward"].mean().item())
        logger.log({"iter": it, "mean_reward": mean_r, "loss_total": total_loss / max(updates, 1)})
        if mean_r > best:
            best = mean_r
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": config}, args.out / "best.pt")
    print(f"best mean reward: {best:.4f}")


if __name__ == "__main__":
    main()
