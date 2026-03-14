"""PyTorch Qwen3TTSTokenizerV1 model."""

import torch
from torch import nn
from transformers.utils import auto_docstring, logging


from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1DecoderDiTConfig,
)
from .modeling_qwen3_tts_tokenizer_v1_bases import (
    Qwen3TTSTokenizerV1DecoderPreTrainedModel,
    Qwen3TTSTokenizerV1EncoderPreTrainedModel,
)
from .modeling_qwen3_tts_tokenizer_v1_bigvgan import (
    Qwen3TTSTokenizerV1DecoderBigVGANModel,
)
from .modeling_qwen3_tts_tokenizer_v1_dit import (
    AdaLayerNormZero_Final,
    DiTCodecEmbedding,
    DiTDecoderLayer,
    DiTInputEmbedding,
    DiTTimestepEmbedding,
    Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding,
)

logger = logging.get_logger(__name__)


# Extracted from modeling_qwen3_tts_tokenizer_v1.py for better navigation.

# Keep shared pretrained bases in core, then import specialized blocks.


@auto_docstring
class Qwen3TTSTokenizerV1DecoderDiTModel(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    config: Qwen3TTSTokenizerV1DecoderDiTConfig
    _no_split_modules = ["DiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        self.text_embed = DiTCodecEmbedding(
            config.num_embeds, config.emb_dim, config.repeats
        )
        self.input_embed = DiTInputEmbedding(config)

        self.rotary_embed = Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(
            config.head_dim
        )

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def _create_block_diff(self, hidden_states):
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = (
            torch.arange(seq_len, device=hidden_states.device) // self.block_size
        )  # [seq_length]

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]
        block_diff = block_j - block_i  # (n, n)

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
    ):
        batch_size = hidden_states.shape[0] * 2
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        # Compute embeddings
        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(
            quantized_code, drop_code=False if apply_cfg else drop_code
        )
        text_embedding_unconditioned = (
            self.text_embed(quantized_code, drop_code=True) if apply_cfg else None
        )

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        # Compute positional encodings
        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output

    def optimized_scale(self, positive_flat, negative_flat):
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm
        return st_star

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        noise_initialization = torch.randn(
            [quantized_code.shape[0], 30000, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
        )
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(
            quantized_code.device
        )
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(
            1, maximum_duration, 1
        )

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)

            return (
                guided_prediction
                + (guided_prediction - null_prediction) * guidance_scale
            )

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time,
            1,
            num_steps,
            device=quantized_code.device,
            dtype=conditioning_vector.dtype,
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (
                torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding
            )

        values = initial_state.clone()
        for t0, t1 in zip(time_embedding[:-1], time_embedding[1:]):
            dt = t1 - t0
            vt = ode_function(t0, values)
            values = values + vt * dt

        generated_mel_spectrogram = values.permute(0, 2, 1)
        return generated_mel_spectrogram


__all__ = [
    "Qwen3TTSTokenizerV1DecoderPreTrainedModel",
    "Qwen3TTSTokenizerV1EncoderPreTrainedModel",
    "Qwen3TTSTokenizerV1DecoderBigVGANModel",
    "Qwen3TTSTokenizerV1DecoderDiTModel",
]
