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


class _CaptureTaskEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.last_task_obs: torch.Tensor | None = None

    def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
        self.last_task_obs = task_obs.detach().clone()
        return torch.zeros(*task_obs.shape[:-1], self.embedding_dim, dtype=task_obs.dtype, device=task_obs.device)


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


def test_transformer_model_multiplies_task_obs_by_mode_mapping() -> None:
    """Mode mapping should be broadcast across task tokens and multiplied into task observations."""
    task_obs = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    obs = TensorDict(
        {
            "policy": torch.ones(1, 2, 5),
            "policy_task": task_obs.clone(),
            "action": torch.zeros(1, 2, NUM_ACTIONS),
            "mode_mapping": torch.tensor([[1.0, 0.0, 2.0]]),
        },
        batch_size=[1],
    )
    actor = TransformerModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        NUM_ACTIONS,
        prop_obs_group="policy",
        task_obs_group="policy_task",
        action_obs_group="action",
        mode_mapping_group="mode_mapping",
        embed_dim=16,
        num_heads=2,
        ff_dim=32,
        num_layers=1,
    )
    capture = _CaptureTaskEmbedder()
    actor.task_embedder = capture

    actor(obs)

    assert capture.last_task_obs is not None
    assert torch.allclose(capture.last_task_obs, task_obs * torch.tensor([[[1.0, 0.0, 2.0]]]))
    assert torch.equal(obs["policy_task"], task_obs)


def test_transformer_model_rejects_mode_mapping_final_dim_mismatch() -> None:
    """Mode mapping final dimension must match the task observation dimension."""
    obs = _make_transformer_obs()
    obs["mode_mapping"] = torch.ones(NUM_ENVS, 6)

    with pytest.raises(ValueError, match="mode_mapping final dimension must match task observation dimension"):
        _make_actor(obs)


def test_transformer_model_accepts_mode_2d_single_token_or_per_task_token() -> None:
    """Mode observations should support [B, D], [B, 1, D], and [B, T, D] layouts."""
    task_tokens = 3
    mode_2d = torch.arange(NUM_ENVS * 2, dtype=torch.float32).reshape(NUM_ENVS, 2)
    mode_single_token = mode_2d.unsqueeze(1)
    mode_per_token = torch.arange(NUM_ENVS * task_tokens * 2, dtype=torch.float32).reshape(NUM_ENVS, task_tokens, 2)
    cases = [
        (mode_2d, mode_2d.unsqueeze(1).expand(-1, task_tokens, -1)),
        (mode_single_token, mode_single_token.expand(-1, task_tokens, -1)),
        (mode_per_token, mode_per_token),
    ]

    for mode, expected_mode in cases:
        obs = _make_transformer_obs(task_tokens=task_tokens)
        obs["mode"] = mode
        actor = _make_actor(obs)
        capture = _CaptureTaskEmbedder()
        actor.task_embedder = capture

        actor(obs)

        assert capture.last_task_obs is not None
        assert torch.equal(capture.last_task_obs[..., -2:], expected_mode)


def test_transformer_model_rejects_invalid_mode_token_length() -> None:
    """Mode token length must be one or match task observation token length."""
    obs = _make_transformer_obs(task_tokens=3)
    obs["mode"] = torch.ones(NUM_ENVS, 2, 2)

    with pytest.raises(ValueError, match="mode token length must be 1 or match task observation token length"):
        _make_actor(obs)


@pytest.mark.parametrize(
    ("missing_group", "match"),
    [
        ("policy_task", "Observation group 'policy_task' is missing"),
        ("action", "Observation group 'action' is missing"),
    ],
)
def test_transformer_model_reports_missing_configured_task_action_groups(missing_group: str, match: str) -> None:
    """Configured task/action group misses should name the missing group."""
    obs = _make_transformer_obs()
    del obs[missing_group]

    with pytest.raises(KeyError, match=match):
        _make_actor(obs)


@pytest.mark.parametrize(
    ("obs", "kwargs", "match"),
    [
        (
            TensorDict(
                {
                    "policy": torch.randn(NUM_ENVS, 5),
                    "action": torch.randn(NUM_ENVS, NUM_ACTIONS),
                },
                batch_size=[NUM_ENVS],
            ),
            {"action_obs_group": "action"},
            "task_obs_group must be provided unless observation group 'task' is present",
        ),
        (
            TensorDict(
                {
                    "policy": torch.randn(NUM_ENVS, 5),
                    "policy_task": torch.randn(NUM_ENVS, 7),
                },
                batch_size=[NUM_ENVS],
            ),
            {"task_obs_group": "policy_task"},
            "action_obs_group must be provided unless observation group 'action' is present",
        ),
    ],
)
def test_transformer_model_reports_missing_default_task_action_groups(
    obs: TensorDict, kwargs: dict[str, str], match: str
) -> None:
    """Default task/action group misses should explain which constructor argument is required."""
    with pytest.raises(ValueError, match=match):
        TransformerModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            prop_obs_group="policy",
            embed_dim=16,
            num_heads=2,
            ff_dim=32,
            num_layers=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "distribution_cfg",
    [
        {"class_name": "HeteroscedasticGaussianDistribution", "init_std": 0.5},
        {"class_name": "BetaDistribution"},
    ],
)
def test_transformer_model_reshapes_two_slice_distribution_inputs(distribution_cfg: dict[str, object]) -> None:
    """Distributions with [2, output_dim] inputs should work through the TransformerModel adapter."""
    torch.manual_seed(0)
    obs = _make_transformer_obs()
    actor = _make_actor(obs, distribution_cfg=distribution_cfg)

    latent = actor.get_latent(obs)
    deterministic = actor(obs)
    sampled = actor(obs, stochastic_output=True)
    log_prob = actor.get_output_log_prob(sampled)

    assert latent.shape == (NUM_ENVS, 2 * NUM_ACTIONS)
    assert actor._reshape_distribution_input(latent).shape == (NUM_ENVS, 2, NUM_ACTIONS)
    assert deterministic.shape == (NUM_ENVS, NUM_ACTIONS)
    assert sampled.shape == (NUM_ENVS, NUM_ACTIONS)
    assert log_prob.shape == (NUM_ENVS,)


def test_transformer_model_normalization_updates_only_prop_sequence_dims() -> None:
    """Observation normalization should flatten 3D prop sequences without including task/action values."""
    obs = _make_transformer_obs(num_envs=2, context=2)
    prop_obs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
            [[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]],
        ]
    )
    obs["policy"] = prop_obs
    obs["policy_task"] = torch.full_like(obs["policy_task"], 10_000.0)
    obs["action"] = torch.full_like(obs["action"], -10_000.0)
    actor = _make_actor(obs, obs_normalization=True)

    actor.update_normalization(obs)

    assert actor.obs_normalizer.count.item() == 4
    assert actor.obs_normalizer.mean.shape == (5,)
    assert torch.allclose(actor.obs_normalizer.mean, prop_obs.reshape(-1, 5).mean(dim=0))
