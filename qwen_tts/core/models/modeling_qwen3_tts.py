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

import json
import os
from collections.abc import Sequence
from typing import Optional, TypedDict

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.utils import logging
from transformers.utils.hub import cached_file

from ...inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from .modeling_qwen3_tts_core import (
    Qwen3TTSPreTrainedModel,
    Qwen3TTSSpeakerEncoder,
    mel_spectrogram,
)
from .modeling_qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerModel,
)

logger = logging.get_logger(__name__)


class VoiceClonePrompt(TypedDict):
    ref_code: list[torch.Tensor | None]
    ref_spk_embedding: list[torch.Tensor]
    x_vector_only_mode: list[bool]
    icl_mode: list[bool]


GenerateConfigPrimitive = str | int | float | bool | None
GenerateConfigValue = (
    GenerateConfigPrimitive
    | list["GenerateConfigValue"]
    | dict[str, "GenerateConfigValue"]
)


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub. Users can specify the
    allow_patterns to download only the necessary weights.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE

    for allow_pattern in allow_patterns:
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_pattern,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_only,
        )
    return hf_folder


class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel):
    config_class = Qwen3TTSConfig

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config

        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)

        if config.tts_model_type == "base":
            self.speaker_encoder: Qwen3TTSSpeakerEncoder | None = (
                Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
            )
        else:
            self.speaker_encoder = None

        self.speech_tokenizer: Qwen3TTSTokenizer | None = None
        self.generate_config: dict[str, GenerateConfigValue] | None = None

        supported_speakers = self.config.talker_config.spk_id or {}
        supported_languages = self.config.talker_config.codec_language_id or {}
        self.supported_speakers = list(supported_speakers.keys())
        self.supported_languages = ["auto"]
        for language_id in supported_languages.keys():
            if "dialect" not in language_id:
                self.supported_languages.append(language_id)

        self.speaker_encoder_sample_rate = (
            self.config.speaker_encoder_config.sample_rate
        )
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type

        self.post_init()

    def load_speech_tokenizer(self, speech_tokenizer: Qwen3TTSTokenizer) -> None:
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(
        self, generate_config: dict[str, GenerateConfigValue]
    ) -> None:
        self.generate_config = generate_config

    def get_supported_speakers(self) -> list[str]:
        return self.supported_speakers

    def get_supported_languages(self) -> list[str]:
        return self.supported_languages

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        # Hotfix to enable passing the correct attn implementation which is stored in the config but not in kwargs
        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if (
            requested_attn_implementation is None
            and config
            and config._attn_implementation
        ):
            requested_attn_implementation = config._attn_implementation

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            attn_implementation=requested_attn_implementation,
            **kwargs,
        )
        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            download_cache_dir = kwargs.get("cache_dir", cache_dir)
            download_revision = kwargs.get("revision", revision)
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=download_cache_dir,
                allow_patterns=["speech_tokenizer/*"],
                revision=download_revision,
            )
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if speech_tokenizer_path is None:
            raise ValueError(
                f"""{pretrained_model_name_or_path}/{speech_tokenizer_path} not exists"""
            )
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            *model_args,
            **kwargs,
        )
        model.load_speech_tokenizer(speech_tokenizer)

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if generate_config_path is None:
            raise ValueError(
                f"{pretrained_model_name_or_path}/generation_config.json not exists"
            )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            generate_config = json.load(f)
        model.load_generate_config(generate_config)

        return model

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        assert sr == 24000, "Only support 24kHz audio"
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mels = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        speaker_encoder = self.speaker_encoder
        if speaker_encoder is None:
            raise RuntimeError("Speaker encoder is not available for this model.")
        speaker_embedding = speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return speaker_embedding

    @torch.inference_mode()
    def generate_speaker_prompt(
        self, voice_clone_prompt: VoiceClonePrompt
    ) -> list[torch.Tensor]:
        voice_clone_spk_embeds: list[torch.Tensor] = []
        for index in range(len(voice_clone_prompt["ref_spk_embedding"])):
            ref_spk_embedding = (
                voice_clone_prompt["ref_spk_embedding"][index]
                .to(self.talker.device)
                .to(self.talker.dtype)
            )
            voice_clone_spk_embeds.append(ref_spk_embedding)

        return voice_clone_spk_embeds

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(
                    self.talker.code_predictor.get_input_embeddings()[i - 1](
                        ref_code[:, i : i + 1]
                    )
                )
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat(
            [
                self.talker.get_input_embeddings()(
                    torch.tensor(
                        [
                            [
                                self.config.talker_config.codec_bos_id,
                            ]
                        ],
                        device=self.talker.device,
                        dtype=text_id.dtype,
                    )
                ),
                codec_embed,
            ],
            dim=1,
        )
        # compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor(
                    [
                        [
                            self.config.talker_config.codec_pad_id,
                        ]
                        * text_lens
                    ],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat(
                [icl_input_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[
                    :, codec_lens:
                ]
            else:
                text_embed = torch.cat(
                    [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1
                )
                return text_embed + codec_embed, tts_pad_embed

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor | None]] = None,
        ref_ids: Optional[list[torch.Tensor | None]] = None,
        voice_clone_prompt: Optional[VoiceClonePrompt] = None,
        languages: Optional[list[str]] = None,
        speakers: Optional[list[str | None]] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs: GenerateConfigPrimitive,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if input_ids is None or len(input_ids) == 0:
            raise ValueError("`input_ids` must be a non-empty list of tensors.")

        batch_size = len(input_ids)
        if languages is None:
            languages = ["auto"] * batch_size
        if len(languages) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, languages={len(languages)}"
            )

        if speakers is None:
            speakers = [None] * batch_size
        if len(speakers) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, speakers={len(speakers)}"
            )

        if instruct_ids is None:
            instruct_ids = [None] * batch_size
        elif len(instruct_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, instruct_ids={len(instruct_ids)}"
            )

        if ref_ids is None:
            ref_ids = [None] * batch_size
        elif len(ref_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: input_ids={batch_size}, ref_ids={len(ref_ids)}"
            )

        if voice_clone_prompt is not None:
            if not (
                len(voice_clone_prompt["ref_code"])
                == len(voice_clone_prompt["ref_spk_embedding"])
                == len(voice_clone_prompt["x_vector_only_mode"])
                == len(voice_clone_prompt["icl_mode"])
                == batch_size
            ):
                raise ValueError(
                    "Batch size mismatch in `voice_clone_prompt` fields and `input_ids`."
                )

        suppress_tokens = [
            i
            for i in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size,
            )
            if i not in (self.config.talker_config.codec_eos_token_id,)
        ]
        _ = kwargs

        talker_input_embeds: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        voice_clone_spk_embeds: list[torch.Tensor] | None = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(instruct_id)
                    )
                )

        trailing_text_hiddens: list[torch.Tensor] = []
        tts_pad_embed_last: torch.Tensor | None = None
        for index, (input_id, language, speaker) in enumerate(
            zip(input_ids, languages, speakers)
        ):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = self.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = self.talker.get_input_embeddings()(
                        torch.tensor(
                            spk_id,
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
            else:
                if voice_clone_prompt is not None and (
                    voice_clone_prompt["x_vector_only_mode"][index]
                    or voice_clone_prompt["icl_mode"][index]
                ):
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            language_lower = language.lower()
            if language_lower == "auto":
                language_id = None
            else:
                if language_lower not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                language_id = self.config.talker_config.codec_language_id[
                    language_lower
                ]

            if (
                language_lower in ["chinese", "auto"]
                and speaker != ""
                and speaker is not None
                and self.config.talker_config.spk_is_dialect[speaker.lower()]
                is not False
            ):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [
                            [
                                self.config.tts_bos_token_id,
                                self.config.tts_eos_token_id,
                                self.config.tts_pad_token_id,
                            ]
                        ],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)
            tts_pad_embed_last = tts_pad_embed

            if language_id is None:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]
            else:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                torch.tensor(
                    codec_prefill_list,
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [
                        [
                            self.config.talker_config.codec_pad_id,
                            self.config.talker_config.codec_bos_id,
                        ]
                    ],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat(
                    [codec_input_emebdding_0, codec_input_emebdding_1], dim=1
                )
            else:
                codec_input_emebdding = torch.cat(
                    [
                        codec_input_emebdding_0,
                        speaker_embed.view(1, 1, -1),
                        codec_input_emebdding_1,
                    ],
                    dim=1,
                )

            _talker_input_embed_role = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = (
                torch.cat(
                    (
                        tts_pad_embed.expand(
                            -1, codec_input_emebdding.shape[1] - 2, -1
                        ),
                        tts_bos_embed,
                    ),
                    dim=1,
                )
                + codec_input_emebdding[:, :-1]
            )
            talker_input_embed = torch.cat(
                (_talker_input_embed_role, _talker_input_embed), dim=1
            )

            ref_code_item = (
                voice_clone_prompt["ref_code"][index]
                if voice_clone_prompt is not None
                else None
            )
            if (
                voice_clone_prompt is not None
                and ref_code_item is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                ref_id = ref_ids[index]
                if ref_id is None:
                    raise ValueError(
                        "`ref_ids` is required for ICL mode voice clone generation."
                    )
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_id[:, 3:-2],
                    ref_code=ref_code_item.to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat(
                    [talker_input_embed, icl_input_embed], dim=1
                )
            else:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        self.talker.text_projection(
                            self.talker.get_text_embeddings()(input_id[:, 3:4])
                        )
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    self.talker.text_projection(
                                        self.talker.get_text_embeddings()(
                                            input_id[:, 3:-5]
                                        )
                                    ),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [
                                        [
                                            self.config.talker_config.codec_pad_id,
                                        ]
                                        * (input_id[:, 3:-5].shape[1] + 1)
                                    ],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [
                                        [
                                            self.config.talker_config.codec_bos_id,
                                        ]
                                    ],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            self.talker.text_projection(
                                self.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        merged_talker_input_embeds: list[torch.Tensor] = []
        for sample_embeds in talker_input_embeds:
            if len(sample_embeds) == 0:
                raise RuntimeError(
                    "Each batch item must contain at least one embed block."
                )
            merged_talker_input_embeds.append(torch.cat(sample_embeds, dim=1))

        if tts_pad_embed_last is None:
            raise RuntimeError("`tts_pad_embed` was not created during generation.")

        original_lengths = torch.tensor(
            [t.shape[1] for t in merged_talker_input_embeds]
        )
        sequences = [t.squeeze(0) for t in merged_talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed, batch_first=True, padding_value=0.0
        )
        talker_input_embeds_tensor = padded_reversed.flip(dims=[1])

        generated_batch_size, max_len = (
            talker_input_embeds_tensor.shape[0],
            talker_input_embeds_tensor.shape[1],
        )
        indices = torch.arange(max_len).expand(generated_batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (
            (indices >= num_pads.unsqueeze(1))
            .long()
            .to(talker_input_embeds_tensor.device)
        )

        pad_embedding_vector = tts_pad_embed_last.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad, batch_first=True, padding_value=0.0
        )
        arange_tensor = torch.arange(
            max(trailing_text_original_lengths), device=padded_hiddens.device
        ).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(
            trailing_text_original_lengths, device=padded_hiddens.device
        ).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens_tensor = padded_hiddens

        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds_tensor,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens_tensor,
            tts_pad_embed=tts_pad_embed_last,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            eos_token_id=(
                eos_token_id
                if eos_token_id is not None
                else self.config.talker_config.codec_eos_token_id
            ),
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

        hidden_states = getattr(talker_result, "hidden_states", None)
        if not isinstance(hidden_states, Sequence):
            raise RuntimeError(
                "Talker generate output does not contain `hidden_states`."
            )

        talker_code_steps: list[torch.Tensor] = []
        talker_hidden_steps: list[torch.Tensor] = []
        for step in hidden_states:
            if not isinstance(step, tuple) or len(step) == 0:
                continue
            codec_ids = step[-1]
            if isinstance(codec_ids, torch.Tensor):
                talker_code_steps.append(codec_ids)

            text_hidden_states = step[0]
            if (
                isinstance(text_hidden_states, tuple)
                and len(text_hidden_states) > 0
                and isinstance(text_hidden_states[-1], torch.Tensor)
            ):
                talker_hidden_steps.append(text_hidden_states[-1][:, -1:])

        if len(talker_code_steps) == 0 or len(talker_hidden_steps) == 0:
            raise RuntimeError(
                "Talker generation returned empty hidden/code states; cannot decode."
            )

        talker_codes = torch.stack(talker_code_steps, dim=1)
        talker_hidden_states = torch.cat(talker_hidden_steps, dim=1)[:, :-1]

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(
            has_stop_token, stop_indices, talker_codes.shape[1]
        )

        talker_codes_list = [
            talker_codes[i, :length] for i, length in enumerate(effective_lengths)
        ]
        talker_hidden_states_list = [
            talker_hidden_states[i, :length, :]
            for i, length in enumerate(effective_lengths)
        ]
        return talker_codes_list, talker_hidden_states_list


__all__ = [
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerModel",
]
