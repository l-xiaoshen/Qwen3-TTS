"""PyTorch Qwen3TTS model."""

from typing import cast

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import (
    BlockMask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging
from typing_extensions import override

from .configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .modeling_qwen3_tts_attention import (
    DecoderLayerOutput,
    Qwen3TTSDecoderLayer,
    Qwen3TTSPreTrainedModel,
    Qwen3TTSRMSNorm,
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerCodePredictorOutputWithPast,
)

logger = logging.get_logger(__name__)
CausalMask = torch.Tensor | BlockMask | None


def _apply_tensor_module(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    return cast(torch.Tensor, module(hidden_states))


class PredictorForwardKwargs(FlashAttentionKwargs, total=False):
    num_items_in_batch: int | torch.Tensor


# Extracted from modeling_qwen3_tts.py for better navigation.


# Extracted from modeling_qwen3_tts_talker.py for better navigation.


class Qwen3TTSTalkerCodePredictorModel(Qwen3TTSPreTrainedModel):
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(
        self, config: Qwen3TTSTalkerCodePredictorConfig, embedding_dim: int
    ) -> None:
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
                nn.Embedding(config.vocab_size, embedding_dim, self.padding_idx)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @override
    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def get_input_embedding(self, index: int) -> nn.Embedding:
        return cast(nn.Embedding, self.codec_embedding[index])

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.codec_embedding = cast(nn.ModuleList, value)

    @override
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        generation_steps: int | None = None,
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
            warning_once = getattr(logger, "warning_once", None)
            if callable(warning_once):
                warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
            else:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
            use_cache = False

        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` is required.")

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
        causal_mask_mapping: dict[str, CausalMask]
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
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
                    )
                )
        hidden_states: torch.Tensor = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.rotary_emb(hidden_states, position_ids),
        )

        # decoder layers
        all_hidden_states: list[torch.Tensor] = []
        all_self_attns: list[torch.Tensor] = []

        for decoder_layer in self.decoder_layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = cast(
                DecoderLayerOutput,
                decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                ),
            )

            if output_attentions:
                hidden_states, self_attn = cast(
                    tuple[torch.Tensor, torch.Tensor | None], layer_outputs
                )
                if self_attn is not None:
                    all_self_attns.append(self_attn)
            else:
                hidden_states = cast(tuple[torch.Tensor], layer_outputs)[0]

        hidden_states = _apply_tensor_module(self.norm, hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, hidden_states),
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(
                cast(tuple[torch.FloatTensor, ...], tuple(all_hidden_states))
                if output_hidden_states
                else None
            ),
            attentions=(
                cast(tuple[torch.FloatTensor, ...], tuple(all_self_attns))
                if output_attentions
                else None
            ),
        )


class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
    Qwen3TTSPreTrainedModel, GenerationMixin
):
    _tied_weights_keys: dict[str, str] = {  # noqa: RUF012
        "lm_head": "model.codec_embedding"
    }
    _tp_plan: dict[str, str] = {  # noqa: RUF012
        "lm_head.*": "colwise_gather_output",
        "model.codec_embedding.*": "embedding_rowwise",
    }
    _pp_plan: dict[str, tuple[list[str], list[str]]] = {  # noqa: RUF012
        "lm_head": (["hidden_states"], ["logits"])
    }
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
    ) -> None:
        super().__init__(config)
        if (
            config.tie_word_embeddings
            and config.hidden_size != talker_config.hidden_size
        ):
            raise ValueError(
                "Tied predictor embeddings require matching predictor and talker hidden sizes."
            )
        self.model: Qwen3TTSTalkerCodePredictorModel = Qwen3TTSTalkerCodePredictorModel(
            config, talker_config.hidden_size
        )
        self.vocab_size = config.vocab_size
        self.lm_head: nn.ModuleList = nn.ModuleList(
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

    @override
    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def get_input_embedding(self, index: int) -> nn.Embedding:
        return self.model.get_input_embedding(index)

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    @override
    def get_output_embeddings(self) -> nn.ModuleList:
        return self.lm_head

    def get_output_embedding(self, index: int) -> nn.Linear:
        return cast(nn.Linear, self.lm_head[index])

    @override
    def set_output_embeddings(self, new_embeddings: nn.ModuleList) -> None:
        self.lm_head = new_embeddings

    @override
    def set_decoder(self, decoder: Qwen3TTSTalkerCodePredictorModel) -> None:
        self.model = decoder

    @override
    def get_decoder(self) -> Qwen3TTSTalkerCodePredictorModel:
        return self.model

    def forward_finetune(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        generation_steps: int | None = None,
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
        inputs_embeds = _apply_tensor_module(
            self.small_to_mtp_projection, inputs_embeds
        )

        flash_attn_kwargs: FlashAttentionKwargs = {
            "cu_seq_lens_q": kwargs.get("cu_seq_lens_q"),
            "cu_seq_lens_k": kwargs.get("cu_seq_lens_k"),
            "max_length_q": kwargs.get("max_length_q"),
            "max_length_k": kwargs.get("max_length_k"),
        }

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
            **flash_attn_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                "Talker predictor output does not contain tensor `last_hidden_state`."
            )

        logit_parts: list[torch.Tensor] = []
        for i in range(1, self.config.num_code_groups):
            logit_parts.append(
                _apply_tensor_module(
                    self.get_output_embedding(i - 1), hidden_states[:, i]
                )
            )
        logits = torch.stack(logit_parts, dim=1)

        loss = None
        if labels is not None:
            loss_kwargs: dict[str, int | torch.Tensor] = {}
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        return Qwen3TTSTalkerCodePredictorOutputWithPast(loss=loss, logits=logits)

    @override
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, CausalMask] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        generation_steps: int | None = None,
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
            inputs_embeds = cast(
                torch.Tensor,
                self.model.get_input_embedding(generation_step_value - 1)(input_ids),
            )
        inputs_embeds = _apply_tensor_module(
            self.small_to_mtp_projection, inputs_embeds
        )

        flash_attn_kwargs: FlashAttentionKwargs = {
            "cu_seq_lens_q": kwargs.get("cu_seq_lens_q"),
            "cu_seq_lens_k": kwargs.get("cu_seq_lens_k"),
            "max_length_q": kwargs.get("max_length_q"),
            "max_length_k": kwargs.get("max_length_k"),
        }

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
            **flash_attn_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                "Talker predictor output does not contain tensor `last_hidden_state`."
            )
        logits = _apply_tensor_module(
            self.get_output_embedding(generation_step_value), hidden_states
        )

        loss = None
        if labels is not None:
            loss_kwargs: dict[str, int | torch.Tensor] = {}
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is not None:
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

    @override
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, object],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, object]:
        model_kwargs = cast(
            dict[str, object],
            super()._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder, num_new_tokens
            ),
        )
        predictor_outputs = cast(Qwen3TTSTalkerCodePredictorOutputWithPast, outputs)
        model_kwargs["generation_steps"] = predictor_outputs.generation_steps
        return model_kwargs
