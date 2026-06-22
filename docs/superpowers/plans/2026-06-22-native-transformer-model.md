# Native TransformerModel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `TransformerModel` to local `rsl_rl` that works as a normal PPO actor or critic without porting old `my_rsl_rl` runner/PPO behavior.

**Architecture:** Add reusable transformer modules under `rsl_rl.modules`, then add `rsl_rl.models.TransformerModel` as a peer of `MLPModel`, not a subclass. The model consumes explicitly configured structured observation groups and uses the current distribution abstraction for stochastic actor output.

**Tech Stack:** Python 3.9+, PyTorch, TensorDict, pytest, local `rsl_rl` model/PPO APIs.

---

## File Structure

- Create `rsl_rl/modules/transformer.py`: reusable transformer building blocks.
- Modify `rsl_rl/modules/__init__.py`: export the transformer building blocks.
- Create `rsl_rl/models/transformer_model.py`: native `TransformerModel` implementation.
- Modify `rsl_rl/models/__init__.py`: export `TransformerModel`.
- Create `tests/models/test_transformer_model.py`: focused unit tests for shape handling, mode handling, distribution behavior, and validation.
- Modify `tests/algorithms/test_ppo.py`: add a small PPO smoke test using transformer actor and critic.
- Modify `docs/guide/configuration.rst`: document `TransformerModel` configuration.
- Modify `docs/api/models.rst`: add `TransformerModel` to model API docs if the existing file lists model classes explicitly.

## Task 1: Add Transformer Building Blocks

**Files:**
- Create: `rsl_rl/modules/transformer.py`
- Modify: `rsl_rl/modules/__init__.py`
- Test: `tests/models/test_transformer_model.py`

- [ ] **Step 1: Write failing module import tests**

Create `tests/models/test_transformer_model.py` with this initial content:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py::test_transformer_building_blocks_are_importable -q
```

Expected: fail with `ImportError` because `HumanoidTransformer`, `RMSNorm`, `SwiGLU`, or `TaskEmbedder` is not exported.

- [ ] **Step 3: Implement transformer building blocks**

Create `rsl_rl/modules/transformer.py` based on the ScaleTrack transformer architecture, with these public classes:

```python
from __future__ import annotations

import math
import torch
import torch.nn as nn


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq)
        return self._apply_rope(x, torch.cos(freqs), torch.sin(freqs))

    def _apply_rope(self, x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor) -> torch.Tensor:
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        cos_freqs = cos_freqs.unsqueeze(0)
        sin_freqs = sin_freqs.unsqueeze(0)
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * cos_freqs - x_odd * sin_freqs
        x_rotated[..., 1::2] = x_even * sin_freqs + x_odd * cos_freqs
        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.silu(self.w(x)) * self.v(x))


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        task_obs_dim: int,
        embedding_dim: int,
        reduced_task_dim: int | None = None,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if reduced_task_dim is not None:
            self.task_projection = self._build_task_projection(task_obs_dim, reduced_task_dim, hidden_dims)
            matrix = torch.randn(embedding_dim, reduced_task_dim, dtype=torch.float)
            q, r = torch.linalg.qr(matrix, mode="reduced")
            diag = torch.sign(torch.diag(r))
            diag[diag == 0] = 1.0
            self.register_buffer("projection_basis", q * diag)
            self._use_reduced_projection = True
        else:
            self.task_projection = self._build_task_projection(task_obs_dim, embedding_dim, hidden_dims)
            self._use_reduced_projection = False

    def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
        task_embedding = self.task_projection(task_obs)
        if not self._use_reduced_projection:
            return task_embedding
        task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(task_embedding, self.projection_basis.T)

    def _build_task_projection(
        self, input_dim: int, output_dim: int, hidden_dims: list[int] | None
    ) -> nn.Module:
        if not hidden_dims:
            return nn.Linear(input_dim, output_dim)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), nn.ELU()]
        for layer_index in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                in_dim = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class HumanoidTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = SwiGLU(embed_dim, ff_dim)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rmsnorm3 = RMSNorm(embed_dim)
        self.cond_norm = RMSNorm(embed_dim)
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(
        self, x: torch.Tensor, task_tokens: torch.Tensor, self_attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_norm = self.rmsnorm1(x)
        x_rope = self.rope(x_norm)
        attn_output, _ = self.self_attention(x_rope, x_rope, x_norm, attn_mask=self_attn_mask)
        x = x + attn_output

        x_norm = self.rmsnorm2(x)
        task_tokens = self.cond_norm(task_tokens)
        cross_output, _ = self.cross_attention(query=x_norm, key=task_tokens, value=task_tokens)
        x = x + cross_output

        return x + self.feed_forward(self.rmsnorm3(x))


class HumanoidTransformer(nn.Module):
    def __init__(
        self,
        prop_obs_dim: int,
        action_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.prop_projection = nn.Linear(prop_obs_dim, embed_dim)
        self.action_projection = nn.Linear(action_dim, embed_dim)
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [HumanoidTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, prop_obs: torch.Tensor, action_obs: torch.Tensor, task_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = prop_obs.shape[0]
        context_size = prop_obs.shape[1]
        prop_tokens = self.prop_projection(prop_obs)
        action_tokens = self.action_projection(action_obs)

        context = prop_tokens.new_empty(batch_size, 2 * context_size - 1, self.embed_dim)
        context[:, 0::2] = prop_tokens
        context[:, 1::2] = action_tokens[:, 1:]

        x = torch.cat([context, self.empty_embedding.expand(batch_size, -1, -1)], dim=1)
        self_attn_mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        row_idx = torch.arange(x.shape[1] - 1, device=x.device)
        col_idx = torch.full((x.shape[1] - 1,), x.shape[1] - 1, device=x.device)
        self_attn_mask[row_idx, col_idx] = True

        for block in self.transformer_blocks:
            x = block(x, task_tokens, self_attn_mask=self_attn_mask)
        x = self.final_norm(x)
        return self.projection_head(x[:, -1, :])

    @torch.no_grad()
    def init_weights(self, head_init_val: torch.Tensor | None = None) -> None:
        num_layers = len(self.transformer_blocks)
        res_scale = 1.0 / math.sqrt(2.0 * num_layers)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                continue
            if isinstance(module, RMSNorm):
                nn.init.ones_(module.scale)
            elif isinstance(module, nn.Linear):
                in_dim = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                embed_dim = module.embed_dim
                std = 1.0 / math.sqrt(embed_dim)
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=std)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=std / math.sqrt(2 * num_layers))
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        for block in self.transformer_blocks:
            block.feed_forward.output.weight.mul_(res_scale)

        nn.init.trunc_normal_(self.empty_embedding, std=0.02)
        nn.init.zeros_(self.projection_head.weight)
        if head_init_val is not None:
            self.projection_head.bias.copy_(head_init_val)
        else:
            nn.init.zeros_(self.projection_head.bias)
```

Modify `rsl_rl/modules/__init__.py` to export:

```python
from .transformer import (
    HumanoidTransformer,
    HumanoidTransformerBlock,
    RMSNorm,
    RoPEPositionalEncoding,
    SwiGLU,
    TaskEmbedder,
)
```

and add those names to `__all__`.

- [ ] **Step 4: Run the import test to verify it passes**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py::test_transformer_building_blocks_are_importable -q
```

Expected: pass.

## Task 2: Add TransformerModel Observation and Forward Behavior

**Files:**
- Create: `rsl_rl/models/transformer_model.py`
- Modify: `rsl_rl/models/__init__.py`
- Modify: `tests/models/test_transformer_model.py`

- [ ] **Step 1: Add failing TransformerModel tests**

Append these tests to `tests/models/test_transformer_model.py`:

```python
from tensordict import TensorDict

from rsl_rl.models import TransformerModel


NUM_ENVS = 4
NUM_ACTIONS = 3


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
```

Also add `import pytest` near the top of the file.

- [ ] **Step 2: Run the new model tests to verify they fail**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py -q
```

Expected: fail with `ImportError` or `NameError` because `TransformerModel` does not exist.

- [ ] **Step 3: Implement TransformerModel**

Create `rsl_rl/models/transformer_model.py` implementing:

- constructor matching the design spec,
- explicit group resolution,
- `_prepare_inputs()` for shape promotion, mode masking, and mode concatenation,
- `forward()` matching `MLPModel` distribution semantics,
- output distribution property methods,
- non-recurrent no-op methods,
- `as_jit()` and `as_onnx()` raising `NotImplementedError`,
- `update_normalization()` updating prop observations only when `obs_normalization=True`.

Use `EmpiricalNormalization`, `HiddenState`, `Distribution`, `HumanoidTransformer`, and `TaskEmbedder` from current `rsl_rl` modules. Do not inherit from `MLPModel`.

Modify `rsl_rl/models/__init__.py`:

```python
from .transformer_model import TransformerModel
```

and add `"TransformerModel"` to `__all__`.

- [ ] **Step 4: Run model tests to verify they pass**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py -q
```

Expected: all tests in `test_transformer_model.py` pass.

## Task 3: Add Mode Adapter Tests and Validation Coverage

**Files:**
- Modify: `tests/models/test_transformer_model.py`
- Modify: `rsl_rl/models/transformer_model.py`

- [ ] **Step 1: Add failing mode handling tests**

Append these tests to `tests/models/test_transformer_model.py`:

```python
class _CaptureTaskEmbedder(torch.nn.Module):
    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.last_task_obs: torch.Tensor | None = None

    def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
        self.last_task_obs = task_obs.detach().clone()
        return torch.zeros(*task_obs.shape[:-1], self.embedding_dim, dtype=task_obs.dtype, device=task_obs.device)


def test_transformer_model_applies_mode_mapping_and_appends_mode() -> None:
    """Mode mapping should mask task features and mode should be appended to every task token."""
    obs = TensorDict(
        {
            "policy": torch.ones(1, 2, 5),
            "policy_task": torch.ones(1, 3, 4),
            "action": torch.zeros(1, 2, NUM_ACTIONS),
            "mode": torch.tensor([[0.25, 0.75]]),
            "mode_mapping": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
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
        mode_group="mode",
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
    expected_task = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 0.25, 0.75]]] * 3).transpose(0, 1)
    assert torch.allclose(capture.last_task_obs, expected_task)


def test_transformer_model_rejects_mode_mapping_dimension_mismatch() -> None:
    """Mode mapping final dimension must match task observation dimension before mode is appended."""
    obs = _make_transformer_obs()
    obs["mode_mapping"] = torch.ones(NUM_ENVS, 6)

    with pytest.raises(ValueError, match="mode_mapping"):
        _make_actor(obs)


def test_transformer_model_rejects_missing_required_group() -> None:
    """Missing configured observation groups should raise a clear error."""
    obs = _make_transformer_obs()
    del obs["policy_task"]

    with pytest.raises(KeyError, match="policy_task"):
        _make_actor(obs)
```

- [ ] **Step 2: Run the mode tests to verify they fail or expose gaps**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py -q
```

Expected: fail if mode handling or validation is missing.

- [ ] **Step 3: Implement missing mode and validation behavior**

Update `TransformerModel` so `_prepare_inputs()`:

- clones `task_obs` before applying `mode_mapping` so input observations are not mutated,
- broadcasts `[B, D]` `mode_mapping` to `[B, T, D]`,
- broadcasts `[B, D]` `mode` to `[B, T, D]`,
- raises `ValueError` for invalid ranks, mismatched context lengths, invalid mode token lengths, and invalid mode mapping dimensions,
- raises `KeyError` with the missing group name when a configured group is absent.

- [ ] **Step 4: Run the model tests to verify they pass**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py -q
```

Expected: all transformer model tests pass.

## Task 4: Add PPO Integration Smoke Test

**Files:**
- Modify: `tests/algorithms/test_ppo.py`

- [ ] **Step 1: Add a failing PPO transformer smoke test**

Append this test to `tests/algorithms/test_ppo.py`:

```python
def make_transformer_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    """Create observations shaped for TransformerModel PPO tests."""
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 3, 5),
            "policy_task": torch.randn(num_envs, 2, 7),
            "critic": torch.randn(num_envs, 3, 6),
            "critic_task": torch.randn(num_envs, 2, 8),
            "action": torch.randn(num_envs, 3, NUM_ACTIONS),
        },
        batch_size=[num_envs],
    )


def test_ppo_update_runs_with_transformer_actor_and_critic() -> None:
    """PPO should run one update with TransformerModel actor and critic through the normal construct path."""
    class Env:
        num_envs = NUM_ENVS
        num_actions = NUM_ACTIONS

    obs = make_transformer_obs()
    cfg = {
        "num_steps_per_env": NUM_STEPS,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "multi_gpu": None,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "schedule": "fixed",
        },
        "actor": {
            "class_name": "TransformerModel",
            "prop_obs_group": "policy",
            "task_obs_group": "policy_task",
            "action_obs_group": "action",
            "embed_dim": 16,
            "num_heads": 2,
            "ff_dim": 32,
            "num_layers": 1,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "TransformerModel",
            "prop_obs_group": "critic",
            "task_obs_group": "critic_task",
            "action_obs_group": "action",
            "embed_dim": 16,
            "num_heads": 2,
            "ff_dim": 32,
            "num_layers": 1,
        },
    }
    ppo = PPO.construct_algorithm(obs, Env(), cfg, "cpu")

    for _ in range(NUM_STEPS):
        actions = ppo.act(obs)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        ppo.process_env_step(obs, rewards, dones, {})

    ppo.compute_returns(obs)
    loss_dict = ppo.update()

    assert set(loss_dict) == {"value", "surrogate", "entropy"}
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
```

- [ ] **Step 2: Run the PPO transformer test to verify it fails before integration is complete**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/algorithms/test_ppo.py::test_ppo_update_runs_with_transformer_actor_and_critic -q
```

Expected: fail if `TransformerModel` is not resolvable or integration behavior is incomplete.

- [ ] **Step 3: Fix integration gaps only if the test exposes them**

If the test fails because `resolve_callable("TransformerModel")` cannot find the model, verify `rsl_rl/models/__init__.py` exports `TransformerModel`.

If the test fails in storage or distribution behavior, fix `TransformerModel` only. Do not change PPO, runner, or storage for this task.

- [ ] **Step 4: Run model and PPO tests**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py tests/algorithms/test_ppo.py::test_ppo_update_runs_with_transformer_actor_and_critic -q
```

Expected: all selected tests pass.

## Task 5: Add Documentation

**Files:**
- Modify: `docs/guide/configuration.rst`
- Modify: `docs/api/models.rst`

- [ ] **Step 1: Add docs for TransformerModel configuration**

Update `docs/guide/configuration.rst` in the model configuration section after `CNNModel` with a `TransformerModel` subsection. Include:

- `class_name: "TransformerModel"`,
- `prop_obs_group`,
- `task_obs_group`,
- `action_obs_group`,
- optional `mode_group`,
- optional `mode_mapping_group`,
- `embed_dim`,
- `num_heads`,
- `ff_dim`,
- `num_layers`,
- `reduced_task_dim`,
- `task_embedder_hidden_dims`,
- `obs_normalization`,
- `distribution_cfg`.

Add a small YAML example:

```yaml
actor:
  class_name: TransformerModel
  prop_obs_group: policy
  task_obs_group: policy_task
  action_obs_group: action
  mode_group: mode
  mode_mapping_group: mode_mapping
  embed_dim: 256
  num_heads: 4
  ff_dim: 256
  num_layers: 4
  distribution_cfg:
    class_name: GaussianDistribution
    init_std: 0.8
critic:
  class_name: TransformerModel
  prop_obs_group: critic
  task_obs_group: critic_task
  action_obs_group: action
  embed_dim: 256
  num_heads: 4
  ff_dim: 256
  num_layers: 4
```

- [ ] **Step 2: Add API doc reference if needed**

Open `docs/api/models.rst`. If it lists model classes explicitly, add `rsl_rl.models.transformer_model.TransformerModel` in the same style as existing model classes. If it uses `automodule` for all models and no explicit class list exists, leave it unchanged and note that in the report.

- [ ] **Step 3: Run docs-adjacent import checks**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py -q
```

Expected: transformer model tests still pass after docs changes.

## Task 6: Final Verification and Cleanup

**Files:**
- Review all modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models/test_transformer_model.py tests/algorithms/test_ppo.py::test_ppo_update_runs_with_transformer_actor_and_critic -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run broader affected tests**

Run:

```bash
uv run --with pytest --with torch --with tensordict --with numpy python -m pytest tests/models tests/algorithms/test_ppo.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Check formatting and lint-sensitive issues**

Run:

```bash
uv run --with ruff python -m ruff check rsl_rl tests
```

Expected: no ruff violations introduced by this change.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: diff contains only transformer model, transformer modules, tests, and docs; `git diff --check` reports no whitespace errors.
