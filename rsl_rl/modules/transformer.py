# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable transformer building blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RoPEPositionalEncoding(nn.Module):
    """Rotary positional encoding for batch-first token sequences."""

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPEPositionalEncoding requires an even dimension")
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding along the sequence dimension."""
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
    """Root mean square normalization."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by the root mean square of the last dimension."""
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated feed-forward projection."""
        return self.output(self.silu(self.w(x)) * self.v(x))


class TaskEmbedder(nn.Module):
    """Project task observations to transformer embedding tokens."""

    def __init__(
        self,
        task_obs_dim: int,
        embedding_dim: int,
        reduced_task_dim: int | None = None,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if reduced_task_dim is not None:
            if reduced_task_dim > embedding_dim:
                raise ValueError("reduced_task_dim must be less than or equal to embedding_dim")
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
        """Embed task observations into task tokens."""
        task_embedding = self.task_projection(task_obs)
        if not self._use_reduced_projection:
            return task_embedding

        task_embedding = task_embedding / (task_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(task_embedding, self.projection_basis.T)

    def _build_task_projection(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None) -> nn.Module:
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
        """Initialize linear projections with ScaleTrack-style normal weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                in_dim = module.weight.size(1)
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class HumanoidTransformerBlock(nn.Module):
    """ScaleTrack transformer block with self-attention and task cross-attention."""

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
        self,
        x: torch.Tensor,
        task_tokens: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention, task cross-attention, and feed-forward residuals."""
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
    """ScaleTrack humanoid transformer head."""

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
        """Run transformer inference and return the final query-token projection."""
        if prop_obs.shape[:2] != action_obs.shape[:2]:
            raise ValueError("prop_obs and action_obs must have the same batch size and context length")

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
        """Initialize transformer weights with ScaleTrack-style scaling."""
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
