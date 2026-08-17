"""PyTorch Qwen3TTSTokenizerV2 model."""

import math
from collections.abc import Callable, Mapping
from typing import Protocol, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    BlockMask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, logging
from transformers.utils.generic import merge_with_config_defaults
from typing_extensions import override

from .configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)

logger = logging.get_logger(__name__)
AttentionMask = torch.Tensor | BlockMask | None
TensorActivation = Callable[[torch.Tensor], torch.Tensor]


class _DecoderAttentionKwargs(FlashAttentionKwargs, total=False):
    position_ids: torch.Tensor | None
    use_cache: bool | None


class _RopeInitFunction(Protocol):
    def __call__(
        self,
        config: Qwen3TTSTokenizerV2DecoderConfig,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, float]: ...


class _AttentionInterface(Protocol):
    def __call__(
        self,
        module: "Qwen3TTSTokenizerV2DecoderAttention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: AttentionMask,
        *,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class _TensorModule(Protocol):
    def __call__(self, tensor: torch.Tensor, /) -> torch.Tensor: ...


def _call_tensor_module(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    return cast(_TensorModule, module)(tensor)


def _get_attention_interface(implementation: str | None) -> _AttentionInterface:
    return cast(_AttentionInterface, ALL_ATTENTION_FUNCTIONS[implementation])


def _get_rope_parameters(
    config: Qwen3TTSTokenizerV2DecoderConfig,
) -> Mapping[str, object]:
    rope_parameters = config.rope_parameters
    if not isinstance(rope_parameters, dict):
        raise TypeError("`config.rope_parameters` must be a dictionary.")
    return rope_parameters


def _compute_default_rope_parameters(
    config: Qwen3TTSTokenizerV2DecoderConfig,
    device: torch.device | None = None,
    **_: object,
) -> tuple[torch.Tensor, float]:
    rope_theta = _get_rope_parameters(config).get("rope_theta")
    if not isinstance(rope_theta, (int, float)):
        raise TypeError("`config.rope_parameters['rope_theta']` must be numeric.")
    head_dim = config.head_dim
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
    )
    return inv_freq, 1.0


# Extracted from modeling_qwen3_tts_tokenizer_v2.py for better navigation.


# Extracted from modeling_qwen3_tts_tokenizer_v2_core.py for better navigation.


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: "Qwen3TTSTokenizerV2DecoderAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@auto_docstring
class Qwen3TTSTokenizerV2DecoderPreTrainedModel(PreTrainedModel):
    config: Qwen3TTSTokenizerV2DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Qwen3TTSTokenizerV2CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - self.padding
        )
        return ideal_length - length

    @override
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(
            hidden_state, (self.padding, extra_padding), mode="constant", value=0
        )
        return _call_tensor_module(self.conv, hidden_state).contiguous()


class Qwen3TTSTokenizerV2CausalTransConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride
        )

        pad = kernel_size - stride
        self.left_pad = 0
        self.right_pad = int(pad)

    @override
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = _call_tensor_module(self.conv, hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerV2ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = Qwen3TTSTokenizerV2CausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    @override
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input = hidden_states

        hidden_states = _call_tensor_module(self.dwconv, hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = _call_tensor_module(self.norm, hidden_states)
        hidden_states = _call_tensor_module(self.pwconv1, hidden_states)
        hidden_states = _call_tensor_module(self.act, hidden_states)
        hidden_states = _call_tensor_module(self.pwconv2, hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = input + hidden_states

        return hidden_states


class Qwen3TTSTokenizerV2DecoderRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    original_inv_freq: torch.Tensor
    compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)

    def __init__(
        self,
        config: Qwen3TTSTokenizerV2DecoderConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_type = _get_rope_parameters(config).get("rope_type")
        if not isinstance(rope_type, str):
            raise TypeError("`config.rope_parameters['rope_type']` must be a string.")
        self.rope_type = rope_type
        self.rope_init_fn: _RopeInitFunction = _compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = cast(
                _RopeInitFunction, ROPE_INIT_FUNCTIONS[self.rope_type]
            )

        rope_device = torch.device("cpu") if device is None else device
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, rope_device)
        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    @override
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerV2DecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx: int
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: AttentionMask,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[_DecoderAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = _call_tensor_module(
            self.q_norm,
            _call_tensor_module(self.q_proj, hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        key_states = _call_tensor_module(
            self.k_norm,
            _call_tensor_module(self.k_proj, hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        value_states = (
            _call_tensor_module(self.v_proj, hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_implementation = self.config._attn_implementation
        if attention_implementation == "eager":
            attention_interface = cast(_AttentionInterface, eager_attention_forward)
        else:
            attention_interface = _get_attention_interface(attention_implementation)

        if attention_interface is eager_attention_forward:
            eager_attention_mask = (
                attention_mask if isinstance(attention_mask, torch.Tensor) else None
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                eager_attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = _call_tensor_module(self.o_proj, attn_output)
        return attn_output, attn_weights


class Qwen3TTSTokenizerV2DecoderMlp(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = cast(TensorActivation, ACT2FN[config.hidden_act])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(_call_tensor_module(self.gate_proj, x))
        up = _call_tensor_module(self.up_proj, x)
        return _call_tensor_module(self.down_proj, gate * up)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3TTSTokenizerV2DecoderRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Qwen3TTSTokenizerV2DecoderRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @override
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @override
    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3TTSTokenizerV2DecoderLayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://huggingface.co/papers/2103.17239).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(
            torch.full((channels,), initial_scale, requires_grad=True)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class Qwen3TTSTokenizerV2DecoderTransformerLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx: int
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTokenizerV2DecoderAttention(config, layer_idx)
        self.mlp = Qwen3TTSTokenizerV2DecoderMlp(config)
        self.input_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)
        self.mlp_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)
        self.attention_type = "sliding_attention"

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: AttentionMask = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = _call_tensor_module(self.input_layernorm, hidden_states)

        # Self Attention
        hidden_states, _ = cast(
            tuple[torch.Tensor, torch.Tensor | None],
            self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            ),
        )
        hidden_states = residual + _call_tensor_module(
            self.self_attn_layer_scale, hidden_states
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = _call_tensor_module(
            self.post_attention_layernorm, hidden_states
        )
        hidden_states = _call_tensor_module(self.mlp, hidden_states)
        hidden_states = residual + _call_tensor_module(
            self.mlp_layer_scale, hidden_states
        )

        return hidden_states


@auto_docstring
class Qwen3TTSTokenizerV2DecoderTransformerModel(
    Qwen3TTSTokenizerV2DecoderPreTrainedModel
):
    _can_record_outputs: dict[str, type[nn.Module]] = {  # noqa: RUF012
        "hidden_states": Qwen3TTSTokenizerV2DecoderTransformerLayer,
        "attentions": Qwen3TTSTokenizerV2DecoderAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig) -> None:
        super().__init__(config)
        transformer_layers = [
            Qwen3TTSTokenizerV2DecoderTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.layers: nn.ModuleList = nn.ModuleList(transformer_layers)
        self.transformer_layers: list[Qwen3TTSTokenizerV2DecoderTransformerLayer] = (
            transformer_layers
        )
        self.norm = Qwen3TTSTokenizerV2DecoderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.window_size = config.sliding_window

        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @override
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | Mapping[str, AttentionMask] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None:
            raise ValueError("input_ids is not expected")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required for tokenizer decoder.")

        inputs_embeds = _call_tensor_module(self.input_proj, inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        causal_mask_mapping: dict[str, AttentionMask]
        if isinstance(attention_mask, Mapping):
            causal_mask_mapping = dict(attention_mask)
        else:
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                ),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(
                        config=self.config,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                    )
                )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.rotary_emb(hidden_states, position_ids),
        )

        for decoder_layer in self.transformer_layers[: self.config.num_hidden_layers]:
            hidden_states = cast(
                torch.Tensor,
                decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                ),
            )

        hidden_states = _call_tensor_module(self.norm, hidden_states)
        hidden_states = _call_tensor_module(self.output_proj, hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )
