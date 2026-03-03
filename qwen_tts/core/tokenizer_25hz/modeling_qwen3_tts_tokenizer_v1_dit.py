# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3TTSTokenizerV1 model."""

import math
from typing import Optional

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging


from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1DecoderBigVGANConfig,
)
from .modeling_qwen3_tts_tokenizer_v1_speaker import ECAPA_TimeDelayNet

logger = logging.get_logger(__name__)


# Extracted from modeling_qwen3_tts_tokenizer_v1_core.py for better navigation.


class Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DiTInputEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        self.spk_encoder = ECAPA_TimeDelayNet(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: Optional[bool] = False,
        code_embed_uncond: Optional[bool] = None,
        apply_cfg: Optional[bool] = True,
    ):
        if apply_cfg:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat(
                [speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0
            )
            condition_vector = torch.cat(
                [condition_vector, torch.zeros_like(condition_vector)], dim=0
            )
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:  # cfg for cond audio
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        condition_vector = (
            self.spk_encoder(condition_vector)
            .unsqueeze(1)
            .repeat(1, hidden_states.size(1), 1)
        )
        hidden_states = self.proj(
            torch.cat(
                (hidden_states, condition_vector, code_embed, speaker_embedding), dim=-1
            )
        )

        return hidden_states


# Transformer backbone using DiT blocks
class DiTCodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )

        hidden_states = (
            self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        hidden_states = (
            self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        )
        return hidden_states


# FeedForward
class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states


# Modified from Llama with a different rotate function, will fixed in next release
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

    def rotate_half_codec(x):
        # x = rearrange(x, "... (d r) -> ... d r", r=2)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)]
        )

    def forward(
        self,
        hidden_states,  # noised input x
        position_embeddings=None,  # rotary position embedding for x
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE, will be fixed at next release
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        attention_weights = attention_weights.reshape(
            batch_size, -1, self.heads * head_dim
        )
        attention_weights = attention_weights.to(query.dtype)

        # linear proj
        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output


# time step conditioning embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, hidden_states, scale=1000):
        device = hidden_states.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * hidden_states.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(hidden_states)


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList(
            [nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]
        )

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)  # b d
        return time_hidden


class DiTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3TTSTokenizerV1DecoderBigVGANConfig,
        look_ahead_block=0,
        look_backward_block=0,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.ff = DiTMLP(
            dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout
        )

    def forward(
        self, hidden_states, timestep, position_embeddings=None, block_diff=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            hidden_states, emb=timestep
        )

        # attention
        attn_output = self.attn(
            hidden_states=norm,
            position_embeddings=position_embeddings,
            attention_mask=(block_diff >= -float(self.look_backward_block))
            & (block_diff <= float(self.look_ahead_block)),
        )

        # process attention output for input x
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = (
            self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states
