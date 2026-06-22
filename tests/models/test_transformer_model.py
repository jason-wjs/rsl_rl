# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TransformerModel and transformer building blocks."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import TransformerModel
from rsl_rl.modules import (
    HumanoidTransformer,
    HumanoidTransformerBlock,
    RMSNorm,
    RoPEPositionalEncoding,
    SwiGLU,
    TaskEmbedder,
)

NUM_ENVS = 4
NUM_ACTIONS = 3


def test_transformer_building_blocks_are_importable() -> None:
    """Transformer building blocks should be available from rsl_rl.modules."""
    rope = RoPEPositionalEncoding(8)
    norm = RMSNorm(8)
    swiglu = SwiGLU(8, 16)
    embedder = TaskEmbedder(task_obs_dim=5, embedding_dim=8)
    block = HumanoidTransformerBlock(embed_dim=8, num_heads=2, ff_dim=16)
    transformer = HumanoidTransformer(
        prop_obs_dim=4,
        action_dim=3,
        output_dim=2,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        num_layers=1,
    )

    x = torch.ones(2, 3, 8)
    task_tokens = torch.ones(2, 4, 8)
    assert rope(x).shape == (2, 3, 8)
    assert norm(x).shape == (2, 3, 8)
    assert swiglu(x).shape == (2, 3, 8)
    assert embedder(torch.ones(2, 4, 5)).shape == (2, 4, 8)
    assert block(x, task_tokens).shape == (2, 3, 8)
    assert transformer(torch.ones(2, 3, 4), torch.zeros(2, 3, 3), task_tokens).shape == (2, 2)


@pytest.mark.parametrize(
    ("prop_obs", "action_obs", "task_tokens", "match"),
    [
        (
            torch.ones(2, 4),
            torch.zeros(2, 3, 3),
            torch.ones(2, 4, 8),
            "prop_obs must be a 3D tensor",
        ),
        (
            torch.ones(2, 3, 4),
            torch.zeros(2, 3),
            torch.ones(2, 4, 8),
            "action_obs must be a 3D tensor",
        ),
        (
            torch.ones(2, 3, 4),
            torch.zeros(2, 3, 3),
            torch.ones(2, 8),
            "task_tokens must be a 3D tensor",
        ),
        (
            torch.ones(2, 3, 4),
            torch.zeros(2, 2, 3),
            torch.ones(2, 4, 8),
            "prop_obs and action_obs must have the same batch size and context length",
        ),
        (
            torch.ones(2, 0, 4),
            torch.zeros(2, 0, 3),
            torch.ones(2, 4, 8),
            "context length must be at least 1",
        ),
        (
            torch.ones(2, 3, 4),
            torch.zeros(2, 3, 3),
            torch.ones(3, 4, 8),
            "task_tokens batch size must match prop_obs batch size",
        ),
        (
            torch.ones(2, 3, 4),
            torch.zeros(2, 3, 3),
            torch.ones(2, 4, 7),
            "task_tokens embedding dimension must equal embed_dim",
        ),
    ],
)
def test_humanoid_transformer_rejects_invalid_forward_shapes(
    prop_obs: torch.Tensor,
    action_obs: torch.Tensor,
    task_tokens: torch.Tensor,
    match: str,
) -> None:
    """HumanoidTransformer should reject malformed forward inputs with clear errors."""
    transformer = HumanoidTransformer(
        prop_obs_dim=4,
        action_dim=3,
        output_dim=2,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        num_layers=1,
    )

    with pytest.raises(ValueError, match=match):
        transformer(prop_obs, action_obs, task_tokens)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"embed_dim": 0}, "embed_dim must be positive"),
        ({"ff_dim": 0}, "ff_dim must be positive"),
        ({"num_heads": 0}, "num_heads must be positive"),
        ({"num_layers": 0}, "num_layers must be at least 1"),
        ({"embed_dim": 7}, "embed_dim must be even"),
        ({"embed_dim": 10, "num_heads": 4}, "embed_dim must be divisible by num_heads"),
    ],
)
def test_humanoid_transformer_rejects_invalid_constructor_config(
    kwargs: dict[str, int],
    match: str,
) -> None:
    """HumanoidTransformer should reject invalid architectural config early."""
    config = {
        "prop_obs_dim": 4,
        "action_dim": 3,
        "output_dim": 2,
        "embed_dim": 8,
        "num_heads": 2,
        "ff_dim": 16,
        "num_layers": 1,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=match):
        HumanoidTransformer(**config)


def _make_transformer_obs(num_envs: int = NUM_ENVS, context: int = 3, task_tokens: int = 2) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, context, 5),
            "policy_task": torch.randn(num_envs, task_tokens, 7),
            "critic": torch.randn(num_envs, context, 6),
            "critic_task": torch.randn(num_envs, task_tokens, 8),
            "action": torch.randn(num_envs, context, NUM_ACTIONS),
            "mode": torch.tensor([[1.0, 0.0]]).repeat(num_envs, 1),
            "mode_mapping": torch.ones(num_envs, 7),
        },
        batch_size=[num_envs],
    )


def _make_actor(obs: TensorDict, **kwargs: object) -> TransformerModel:
    defaults = {
        "prop_obs_group": "policy",
        "task_obs_group": "policy_task",
        "action_obs_group": "action",
        "mode_group": "mode",
        "mode_mapping_group": "mode_mapping",
        "embed_dim": 16,
        "num_heads": 2,
        "ff_dim": 32,
        "num_layers": 1,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8},
    }
    defaults.update(kwargs)
    return TransformerModel(obs, {"actor": ["policy"], "critic": ["critic"]}, "actor", NUM_ACTIONS, **defaults)


def _make_critic(obs: TensorDict, **kwargs: object) -> TransformerModel:
    defaults = {
        "prop_obs_group": "critic",
        "task_obs_group": "critic_task",
        "action_obs_group": "action",
        "embed_dim": 16,
        "num_heads": 2,
        "ff_dim": 32,
        "num_layers": 1,
    }
    defaults.update(kwargs)
    return TransformerModel(obs, {"actor": ["policy"], "critic": ["critic"]}, "critic", 1, **defaults)


def test_transformer_model_actor_returns_deterministic_distribution_output() -> None:
    """Actor TransformerModel should return the distribution mean by default."""
    obs = _make_transformer_obs()
    actor = _make_actor(obs)

    deterministic = actor(obs)
    sampled = actor(obs, stochastic_output=True)

    assert deterministic.shape == (NUM_ENVS, NUM_ACTIONS)
    assert sampled.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.output_mean.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.output_std.shape == (NUM_ENVS, NUM_ACTIONS)
    assert actor.output_entropy.shape == (NUM_ENVS,)
    assert torch.allclose(deterministic, actor.output_mean, atol=1e-6)


def test_transformer_model_critic_returns_value() -> None:
    """Critic TransformerModel should return raw value output without a distribution."""
    obs = _make_transformer_obs()
    critic = _make_critic(obs)

    value = critic(obs)

    assert value.shape == (NUM_ENVS, 1)


def test_transformer_model_promotes_2d_observations() -> None:
    """2D prop, task, and action observations should be promoted to one-token 3D inputs."""
    obs = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, 5),
            "policy_task": torch.randn(NUM_ENVS, 7),
            "action": torch.randn(NUM_ENVS, NUM_ACTIONS),
        },
        batch_size=[NUM_ENVS],
    )
    actor = TransformerModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        NUM_ACTIONS,
        prop_obs_group="policy",
        task_obs_group="policy_task",
        action_obs_group="action",
        embed_dim=16,
        num_heads=2,
        ff_dim=32,
        num_layers=1,
    )

    assert actor(obs).shape == (NUM_ENVS, NUM_ACTIONS)


def test_transformer_model_rejects_mismatched_context_lengths() -> None:
    """Prop and action context lengths must match."""
    obs = _make_transformer_obs(context=3)
    obs["action"] = torch.randn(NUM_ENVS, 2, NUM_ACTIONS)

    with pytest.raises(ValueError, match="same context length"):
        _make_actor(obs)
