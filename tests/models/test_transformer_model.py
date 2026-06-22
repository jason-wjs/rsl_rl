# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TransformerModel and transformer building blocks."""

from __future__ import annotations

import pytest
import torch

from rsl_rl.modules import (
    HumanoidTransformer,
    HumanoidTransformerBlock,
    RMSNorm,
    RoPEPositionalEncoding,
    SwiGLU,
    TaskEmbedder,
)


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
