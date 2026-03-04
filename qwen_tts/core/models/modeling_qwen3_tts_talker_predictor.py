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
"""PyTorch Qwen3TTS model."""

from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import BlockMask
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging

from .configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .modeling_qwen3_tts_attention import (
    Qwen3TTSDecoderLayer,
    Qwen3TTSPreTrainedModel,
    Qwen3TTSRMSNorm,
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerCodePredictorOutputWithPast,
)

logger = logging.get_logger(__name__)
CausalMask = torch.Tensor | BlockMask | None


class PredictorForwardKwargs(FlashAttentionKwargs, total=False):
    num_items_in_batch: int | torch.Tensor


# Extracted from modeling_qwen3_tts.py for better navigation.


# Extracted from modeling_qwen3_tts_talker.py for better navigation.


class Qwen3TTSTalkerCodePredictorModel(Qwen3TTSPreTrainedModel):
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, embedding_dim: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        decoder_layers = [
            Qwen3TTSDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.layers: nn.ModuleList = nn.ModuleList(decoder_layers)
        self.decoder_layers: list[Qwen3TTSDecoderLayer] = decoder_layers
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.gradient_checkpointing: bool = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, embedding_dim)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        generation_steps: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            if callable(getattr(logger, "warning_once", None)):
                getattr(logger, "warning_once")(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
            else:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        causal_mask_mapping: dict[str, CausalMask]
        if isinstance(attention_mask, dict):
            causal_mask_mapping = {}
            for mask_key, mask_value in attention_mask.items():
                if isinstance(mask_key, str) and (
                    mask_value is None
                    or isinstance(mask_value, (torch.Tensor, BlockMask))
                ):
                    causal_mask_mapping[mask_key] = mask_value
        else:
            attention_mask_tensor = (
                attention_mask if isinstance(attention_mask, torch.Tensor) else None
            )
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask_tensor,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                ),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(
                        config=self.config,
                        input_embeds=inputs_embeds,
                        attention_mask=attention_mask_tensor,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                    )
                )
        hidden_states: torch.FloatTensor = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states: list[torch.FloatTensor] = []
        all_self_attns: list[torch.FloatTensor] = []

        for decoder_layer in self.decoder_layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                self_attn = layer_outputs[1] if len(layer_outputs) > 1 else None
                if self_attn is not None:
                    all_self_attns.append(self_attn)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_self_attns) if output_attentions else None,
        )


class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
    Qwen3TTSPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
    ):
        super().__init__(config)
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(
                talker_config.hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = torch.nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward_finetune(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        generation_steps: Optional[int] = None,
        **kwargs: Unpack[PredictorForwardKwargs],
    ) -> Qwen3TTSTalkerCodePredictorOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required in `forward_finetune`.")
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        flash_attn_kwargs: FlashAttentionKwargs = {}
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is None or isinstance(cu_seq_lens_q, torch.Tensor):
            flash_attn_kwargs["cu_seq_lens_q"] = cu_seq_lens_q
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k")
        if cu_seq_lens_k is None or isinstance(cu_seq_lens_k, torch.Tensor):
            flash_attn_kwargs["cu_seq_lens_k"] = cu_seq_lens_k
        max_length_q = kwargs.get("max_length_q")
        if max_length_q is None or isinstance(max_length_q, int):
            flash_attn_kwargs["max_length_q"] = max_length_q
        max_length_k = kwargs.get("max_length_k")
        if max_length_k is None or isinstance(max_length_k, int):
            flash_attn_kwargs["max_length_k"] = max_length_k

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logits = []
        for i in range(1, self.config.num_code_groups):
            logits.append(self.lm_head[i - 1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss_kwargs: dict[str, int | torch.Tensor] = {}
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if isinstance(num_items_in_batch, (int, torch.Tensor)):
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        return Qwen3TTSTalkerCodePredictorOutputWithPast(loss=loss, logits=logits)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        generation_steps: Optional[int] = None,
        **kwargs: Unpack[PredictorForwardKwargs],
    ) -> Qwen3TTSTalkerCodePredictorOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        generation_step_value: int
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step_value = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            if generation_steps is None:
                raise ValueError("`generation_steps` is required in generation stage.")
            generation_step_value = generation_steps
            if input_ids is None:
                raise ValueError("`input_ids` is required in generation stage.")
            inputs_embeds = self.model.get_input_embeddings()[
                generation_step_value - 1
            ](input_ids)
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        flash_attn_kwargs: FlashAttentionKwargs = {}
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is None or isinstance(cu_seq_lens_q, torch.Tensor):
            flash_attn_kwargs["cu_seq_lens_q"] = cu_seq_lens_q
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k")
        if cu_seq_lens_k is None or isinstance(cu_seq_lens_k, torch.Tensor):
            flash_attn_kwargs["cu_seq_lens_k"] = cu_seq_lens_k
        max_length_q = kwargs.get("max_length_q")
        if max_length_q is None or isinstance(max_length_q, int):
            flash_attn_kwargs["max_length_q"] = max_length_q
        max_length_k = kwargs.get("max_length_k")
        if max_length_k is None or isinstance(max_length_k, int):
            flash_attn_kwargs["max_length_k"] = max_length_k

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_step_value](hidden_states)

        loss = None
        if labels is not None:
            loss_kwargs: dict[str, int | torch.Tensor] = {}
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if isinstance(num_items_in_batch, (int, torch.Tensor)):
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_step_value + 1,
        )

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs
