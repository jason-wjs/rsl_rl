# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TransformerModel and transformer building blocks."""

from __future__ import annotations

import torch

from rsl_rl.modules import HumanoidTransformer, RMSNorm, SwiGLU, TaskEmbedder


def test_transformer_building_blocks_are_importable() -> None:
    """Transformer building blocks should be available from rsl_rl.modules."""
    norm = RMSNorm(8)
    swiglu = SwiGLU(8, 16)
    embedder = TaskEmbedder(task_obs_dim=5, embedding_dim=8)
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
    assert norm(x).shape == (2, 3, 8)
    assert swiglu(x).shape == (2, 3, 8)
    assert embedder(torch.ones(2, 4, 5)).shape == (2, 4, 8)
    assert transformer(torch.ones(2, 3, 4), torch.zeros(2, 3, 3), torch.ones(2, 4, 8)).shape == (2, 2)
