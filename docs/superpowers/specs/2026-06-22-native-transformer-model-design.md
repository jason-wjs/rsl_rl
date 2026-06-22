# Native TransformerModel Design

## Goal

Add a native transformer model to the current `rsl_rl` architecture so transformer
policies can be configured as normal PPO actor and critic models.

The integration should follow the current `rsl_rl` model interface. It should not
port the old `my_rsl_rl` runner, PPO implementation, training-time evaluation,
adaptive motion sampling, old checkpoint format, or multi-GPU advantage
normalization behavior.

## Current State

The local `rsl_rl` repository is at version 5.4.1 and uses a modern split between
algorithm, actor model, critic model, storage, and distribution modules.

The current PPO path constructs actor and critic independently from `cfg["actor"]`
and `cfg["critic"]`. Models are expected to implement the same runtime interface
as `MLPModel`: `forward()`, distribution properties for stochastic actor output,
normalization hooks, recurrent-state no-ops when non-recurrent, and export hooks.

`MLPModel` only supports 2D observations and concatenates configured observation
groups. This is not a good fit for the ScaleTrack humanoid transformer, which
uses structured context observations:

- proprioceptive context tokens,
- action-history context tokens,
- task tokens,
- optional mode masks and mode indicators.

## Scope

Build the first native transformer model with these capabilities:

- Add `rsl_rl.models.TransformerModel`.
- Add reusable transformer building blocks in `rsl_rl.modules.transformer`.
- Support actor usage with `distribution_cfg`.
- Support critic usage without `distribution_cfg`.
- Support 2D observations by promoting them to a single-token 3D context.
- Support 3D context observations directly.
- Support optional task masking through `mode_mapping_group`.
- Support optional mode token concatenation through `mode_group`.
- Integrate through existing `PPO.construct_algorithm()` without runner, PPO, or
  storage changes.
- Add focused unit and PPO integration tests.

## Non-Goals

Do not implement these in the first version:

- `ActorCriticHumanoidTransformer` compatibility.
- `ActorCriticHumanoidTransformerAdapt`.
- Old `my_rsl_rl` checkpoint loading or conversion.
- Motion evaluation during training.
- Adaptive motion sampling.
- Actor/critic split learning rates.
- Multi-GPU global advantage normalization.
- TorchScript or ONNX export support for transformer models.

## Architecture

`TransformerModel` is a peer of `MLPModel`, `RNNModel`, and `CNNModel`, not a
subclass of `MLPModel`.

It implements the current model protocol directly, because it needs to consume
structured 3D observations instead of flat 2D concatenated observations. It may
reuse `Distribution`, `EmpiricalNormalization`, and common no-op recurrent-state
behavior, but it must not reuse `MLPModel`'s observation concatenation or MLP
head path.

The transformer model reads its observation groups explicitly:

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
  task_embedder_hidden_dims: []
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

`obs_set` remains part of the constructor signature for compatibility with
`PPO.construct_algorithm()`, but transformer input semantics are governed by
explicit group fields.

## Model Interface

The constructor should follow this shape:

```python
TransformerModel(
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
)
```

Default group resolution should be conservative:

- `prop_obs_group` defaults to the first group in `obs_groups[obs_set]`.
- `task_obs_group` and `action_obs_group` must be explicitly provided unless
  groups named `"task"` and `"action"` are available.
- `mode_group` and `mode_mapping_group` are optional.

The model must expose the same output properties as `MLPModel` when
`distribution_cfg` is configured:

- `output_mean`
- `output_std`
- `output_entropy`
- `output_distribution_params`
- `get_output_log_prob()`
- `get_kl_divergence()`

It should be non-recurrent:

- `is_recurrent = False`
- `reset()` no-op
- `get_hidden_state()` returns `None`
- `detach_hidden_state()` no-op

`as_jit()` and `as_onnx()` should raise `NotImplementedError` in the first
version with a clear message.

## Observation Adapter

The adapter converts configured observation groups into transformer inputs.

Supported shapes:

- `prop_obs`: `[B, C, D_prop]` or `[B, D_prop]`
- `task_obs`: `[B, T, D_task]` or `[B, D_task]`
- `action_obs`: `[B, C, D_action]` or `[B, D_action]`
- `mode`: optional `[B, D_mode]` or `[B, T, D_mode]`
- `mode_mapping`: optional `[B, D_task]` or `[B, T, D_task]`

Rules:

- 2D tensors are promoted to 3D by inserting a singleton context dimension.
- `mode_mapping` is broadcast across task-token length when needed and multiplied
  into `task_obs`.
- `mode` is broadcast across task-token length when needed and concatenated to
  the last dimension of `task_obs`.
- Training and inference use the same mode and mode-mapping behavior.

The model should validate:

- configured observation group names exist in the input `TensorDict`,
- required observation tensors are 2D or 3D,
- `prop_obs` and `action_obs` have the same context length,
- `mode_mapping` final dimension matches original task-observation dimension,
- `mode` token length is either 1 or matches task-token length.

## Transformer Blocks

Move the transformer building blocks from `ScaleTrack/src/scaletrack_rl` into a
clean `rsl_rl` module namespace:

- `RMSNorm`
- `RoPEPositionalEncoding`
- `SwiGLU`
- `TaskEmbedder`
- `HumanoidTransformerBlock`
- `HumanoidTransformer`

The first implementation can keep the architecture equivalent to the current
ScaleTrack transformer:

- project proprioceptive observations,
- project action observations,
- interleave prop and action context as `[prop_0, action_1, prop_1, ...]`,
- append a learnable empty query token,
- block future attention from context tokens to the query token using a boolean
  self-attention mask,
- cross-attend from context/query tokens to task tokens,
- read the final query token through a projection head.

## Distribution Behavior

`TransformerModel.forward(obs, stochastic_output=False)` computes the transformer
head output.

If `distribution_cfg` is provided:

- `stochastic_output=True` updates the distribution and returns a sample.
- `stochastic_output=False` returns the deterministic distribution output.

If `distribution_cfg` is absent, `forward()` returns the raw transformer output.

This mirrors `MLPModel` behavior while keeping the transformer architecture
separate.

## Testing

Add tests in `tests/models/test_transformer_model.py` covering:

- deterministic actor output shape,
- stochastic actor output and distribution properties,
- critic output shape,
- 2D observation promotion,
- 3D observation handling,
- mode mapping masking,
- mode concatenation,
- clear validation errors for invalid shapes and mismatched context lengths.

Add PPO integration coverage to `tests/algorithms/test_ppo.py` or a focused
transformer-specific test file:

- construct PPO with actor and critic as `TransformerModel`,
- run a small `act -> process_env_step -> compute_returns -> update` smoke test.

The first version does not need full runner training tests or export tests.
