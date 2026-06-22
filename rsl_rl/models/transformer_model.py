# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, HumanoidTransformer, TaskEmbedder
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class TransformerModel(nn.Module):
    """Transformer-based neural model for structured humanoid observations."""

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        prop_obs_group: str | None = None,
        task_obs_group: str | None = None,
        action_obs_group: str | None = None,
        mode_group: str | None = None,
        mode_mapping_group: str | None = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
        reduced_task_dim: int | None = None,
        task_embedder_hidden_dims: list[int] | None = None,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the transformer model."""
        super().__init__()

        self.prop_obs_group = self._resolve_prop_obs_group(obs_groups, obs_set, prop_obs_group)
        self.task_obs_group = self._resolve_required_obs_group(obs, task_obs_group, "task", "task_obs_group")
        self.action_obs_group = self._resolve_required_obs_group(obs, action_obs_group, "action", "action_obs_group")
        self.mode_group = mode_group
        self.mode_mapping_group = mode_mapping_group
        self.obs_groups = self._get_active_obs_groups()

        prop_obs, task_obs, action_obs = self._prepare_inputs(obs, normalize_prop=False)
        self.prop_obs_dim = prop_obs.shape[-1]
        self.task_obs_dim = task_obs.shape[-1]
        self.action_obs_dim = action_obs.shape[-1]
        self.obs_dim = self.prop_obs_dim

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.prop_obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            self._distribution_input_shape = self._resolve_distribution_input_shape(self.distribution.input_dim)
            transformer_output_dim = self._flatten_shape(self._distribution_input_shape)
        else:
            self.distribution = None
            self._distribution_input_shape = None
            transformer_output_dim = output_dim

        self.task_embedder = TaskEmbedder(
            self.task_obs_dim,
            embed_dim,
            reduced_task_dim=reduced_task_dim,
            hidden_dims=task_embedder_hidden_dims,
        )
        self.transformer = HumanoidTransformer(
            self.prop_obs_dim,
            self.action_obs_dim,
            transformer_output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
        )
        self.task_embedder.init_weights()
        self.transformer.init_weights()

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self._make_distribution_head_proxy())

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run transformer inference and apply distribution semantics when configured."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        transformer_output = self.get_latent(obs, masks, hidden_state)

        if self.distribution is not None:
            distribution_input = self._reshape_distribution_input(transformer_output)
            if stochastic_output:
                self.distribution.update(distribution_input)
                return self.distribution.sample()
            return self.distribution.deterministic_output(distribution_input)
        return transformer_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Return the raw transformer head output before distribution handling."""
        prop_obs, task_obs, action_obs = self._prepare_inputs(obs)
        task_tokens = self.task_embedder(task_obs)
        return self.transformer(prop_obs, action_obs, task_tokens)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for TransformerModel)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        if self.distribution is None:
            raise AttributeError("TransformerModel has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        raise NotImplementedError("TransformerModel does not support Torch JIT export yet.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        raise NotImplementedError("TransformerModel does not support ONNX export yet.")

    def update_normalization(self, obs: TensorDict) -> None:
        """Update prop-observation normalization statistics from a batch of observations."""
        if self.obs_normalization:
            prop_obs = self._promote_to_sequence(self._get_obs(obs, self.prop_obs_group), self.prop_obs_group)
            self._validate_feature_dim(prop_obs, self.prop_obs_dim, self.prop_obs_group)
            self.obs_normalizer.update(prop_obs.reshape(-1, prop_obs.shape[-1]))  # type: ignore

    def _prepare_inputs(
        self, obs: TensorDict, normalize_prop: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prop_obs = self._promote_to_sequence(self._get_obs(obs, self.prop_obs_group), self.prop_obs_group)
        task_obs = self._promote_to_sequence(self._get_obs(obs, self.task_obs_group), self.task_obs_group)
        action_obs = self._promote_to_sequence(self._get_obs(obs, self.action_obs_group), self.action_obs_group)

        if prop_obs.shape[:2] != action_obs.shape[:2]:
            raise ValueError("prop_obs and action_obs must have the same context length and batch size")
        if task_obs.shape[0] != prop_obs.shape[0]:
            raise ValueError("task_obs batch size must match prop_obs batch size")

        self._validate_known_feature_dims(prop_obs, task_obs, action_obs)

        if self.mode_mapping_group is not None:
            mode_mapping = self._prepare_token_aligned_obs(
                obs,
                self.mode_mapping_group,
                task_obs.shape[0],
                task_obs.shape[1],
                label="mode_mapping",
            )
            if mode_mapping.shape[-1] != task_obs.shape[-1]:
                raise ValueError("mode_mapping final dimension must match task observation dimension")
            task_obs = task_obs * mode_mapping

        if self.mode_group is not None:
            mode = self._prepare_token_aligned_obs(
                obs,
                self.mode_group,
                task_obs.shape[0],
                task_obs.shape[1],
                label="mode",
            )
            task_obs = torch.cat([task_obs, mode], dim=-1)

        if hasattr(self, "task_obs_dim"):
            self._validate_feature_dim(task_obs, self.task_obs_dim, self.task_obs_group)

        if normalize_prop:
            prop_obs = self.obs_normalizer(prop_obs)
        return prop_obs, task_obs, action_obs

    def _prepare_token_aligned_obs(
        self, obs: TensorDict, group: str, batch_size: int, token_count: int, label: str
    ) -> torch.Tensor:
        tensor = self._promote_to_sequence(self._get_obs(obs, group), group)
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{label} batch size must match task observation batch size")
        if tensor.shape[1] == 1:
            return tensor.expand(-1, token_count, -1)
        if tensor.shape[1] != token_count:
            raise ValueError(f"{label} token length must be 1 or match task observation token length")
        return tensor

    def _promote_to_sequence(self, tensor: torch.Tensor, group: str) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1)
        if tensor.dim() == 3:
            return tensor
        raise ValueError(f"Observation group '{group}' must be a 2D or 3D tensor, got shape {tensor.shape}.")

    def _get_obs(self, obs: TensorDict, group: str) -> torch.Tensor:
        try:
            return obs[group]
        except KeyError as exc:
            raise KeyError(f"Observation group '{group}' is missing.") from exc

    def _resolve_prop_obs_group(
        self, obs_groups: dict[str, list[str]], obs_set: str, prop_obs_group: str | None
    ) -> str:
        if prop_obs_group is not None:
            return prop_obs_group
        try:
            active_obs_groups = obs_groups[obs_set]
        except KeyError as exc:
            raise KeyError(f"Observation set '{obs_set}' is missing from obs_groups.") from exc
        if not active_obs_groups:
            raise ValueError(f"Observation set '{obs_set}' must contain at least one observation group.")
        return active_obs_groups[0]

    def _resolve_required_obs_group(
        self, obs: TensorDict, obs_group: str | None, default_group: str, arg_name: str
    ) -> str:
        if obs_group is not None:
            return obs_group
        if default_group in obs.keys():
            return default_group
        raise ValueError(f"{arg_name} must be provided unless observation group '{default_group}' is present.")

    def _get_active_obs_groups(self) -> list[str]:
        obs_groups = [self.prop_obs_group, self.task_obs_group, self.action_obs_group]
        if self.mode_group is not None:
            obs_groups.append(self.mode_group)
        if self.mode_mapping_group is not None:
            obs_groups.append(self.mode_mapping_group)
        return obs_groups

    def _validate_known_feature_dims(
        self, prop_obs: torch.Tensor, task_obs: torch.Tensor, action_obs: torch.Tensor
    ) -> None:
        if hasattr(self, "prop_obs_dim"):
            self._validate_feature_dim(prop_obs, self.prop_obs_dim, self.prop_obs_group)
        if hasattr(self, "action_obs_dim"):
            self._validate_feature_dim(action_obs, self.action_obs_dim, self.action_obs_group)
        if hasattr(self, "task_obs_dim") and self.mode_group is None:
            self._validate_feature_dim(task_obs, self.task_obs_dim, self.task_obs_group)

    def _validate_feature_dim(self, tensor: torch.Tensor, expected_dim: int, group: str) -> None:
        if tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"Observation group '{group}' final dimension must be {expected_dim}, got {tensor.shape[-1]}."
            )

    def _resolve_distribution_input_shape(self, input_dim: int | list[int] | tuple[int, ...]) -> int | tuple[int, ...]:
        if isinstance(input_dim, int):
            return input_dim
        if isinstance(input_dim, (list, tuple)):
            return tuple(input_dim)
        raise TypeError(f"Unsupported distribution input_dim type: {type(input_dim)}")

    def _flatten_shape(self, shape: int | tuple[int, ...]) -> int:
        if isinstance(shape, int):
            return shape
        return math.prod(shape)

    def _reshape_distribution_input(self, transformer_output: torch.Tensor) -> torch.Tensor:
        if self._distribution_input_shape is None or isinstance(self._distribution_input_shape, int):
            return transformer_output
        return transformer_output.reshape(*transformer_output.shape[:-1], *self._distribution_input_shape)

    def _make_distribution_head_proxy(self) -> nn.Sequential:
        if self._distribution_input_shape is None or isinstance(self._distribution_input_shape, int):
            return nn.Sequential(self.transformer.projection_head)
        return nn.Sequential(
            self.transformer.projection_head,
            nn.Unflatten(dim=-1, unflattened_size=self._distribution_input_shape),
        )
