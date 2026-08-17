# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
import torch

from qwen_tts.core import SpeakerConfiguration, SubTalkerConfiguration

from ..core.models import Qwen3TTSCustomVoiceForConditionalGeneration
from .qwen3_tts_base_model import (
    AudioLike,
    Qwen3TTSBaseModel,
    TTSBatchInput,
    TTSInput,
    TTSInputItem,
)


@dataclass
class CustomVoicePromptItem:
    """
    Container for one sample's reference prompt used by the CustomVoice model.
    """

    ref_code: torch.Tensor  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_text: str = ""


class CustomVoicePromptDict(TypedDict):
    ref_code: list[torch.Tensor]
    ref_text: list[str]


class CustomVoicePromptSingleDict(TypedDict):
    ref_code: torch.Tensor
    ref_text: str


CustomVoicePromptInput = Mapping[str, Sequence[torch.Tensor | str]]
CustomVoicePromptSingleInput = Mapping[str, torch.Tensor | str]
AudioBatchInput = list[AudioLike] | tuple[AudioLike, ...]
StringBatchInput = list[str] | tuple[str, ...]
SpeakerBatchInput = (
    list[SpeakerConfiguration | torch.Tensor]
    | tuple[SpeakerConfiguration | torch.Tensor, ...]
)


class Qwen3TTSCustomVoiceModel(Qwen3TTSBaseModel):
    model: Qwen3TTSCustomVoiceForConditionalGeneration

    def _validate_chunk_instruction_support(
        self, chunks: Sequence[TTSInputItem]
    ) -> None:
        if self.model.tts_model_size == "0b6" and any(
            chunk["instruction"] != "" for chunk in chunks
        ):
            raise ValueError(
                "CustomVoice 0.6B does not support non-empty chunk instructions."
            )

    def _tokenize_custom_voice_chunks(
        self, chunks: Sequence[TTSInputItem]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        input_ids, instruct_ids = self._tokenize_tts_chunks(chunks)
        if self.model.tts_model_size == "0b6":
            instruct_ids = [cast(torch.Tensor | None, None) for _ in range(len(chunks))]
        return input_ids, instruct_ids

    @torch.inference_mode()
    def create_custom_voice_prompt(
        self,
        ref_audio: AudioBatchInput,
        ref_text: StringBatchInput,
    ) -> list[CustomVoicePromptItem]:
        """
        Build reusable ICL prompt items from reference audio/text for the
        CustomVoice model. Speaker identity must still be supplied separately
        through `speaker` during generation.
        """
        self._ensure_model_type("custom_voice", "create_custom_voice_prompt")

        speech_tokenizer = self._require_speech_tokenizer()
        ref_audio_list: list[AudioLike] = list(ref_audio)

        ref_text_list = list(ref_text)
        for item in ref_text_list:
            if item.strip() == "":
                raise ValueError(
                    "`ref_text` items must be non-empty for CustomVoice ICL prompting."
                )

        if len(ref_audio_list) != len(ref_text_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: list[np.ndarray] = []
        ref_sr_for_code: list[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        return [
            CustomVoicePromptItem(ref_code=ref_code, ref_text=ref_text_value)
            for ref_code, ref_text_value in zip(ref_codes, ref_text_list)
        ]

    def _prompt_items_to_custom_voice_prompt(
        self, items: list[CustomVoicePromptItem]
    ) -> CustomVoicePromptDict:
        return CustomVoicePromptDict(
            ref_code=[item.ref_code for item in items],
            ref_text=[item.ref_text for item in items],
        )

    def _prompt_item_to_custom_voice_prompt_single(
        self, item: CustomVoicePromptItem
    ) -> CustomVoicePromptSingleDict:
        return CustomVoicePromptSingleDict(
            ref_code=item.ref_code,
            ref_text=item.ref_text,
        )

    def _coerce_custom_voice_prompt_dict(
        self, prompt: CustomVoicePromptInput
    ) -> CustomVoicePromptDict:
        required_keys = ("ref_code", "ref_text")
        for key in required_keys:
            if key not in prompt:
                raise KeyError(f"Missing key `{key}` in `custom_voice_prompt`.")

        ref_code_raw = prompt["ref_code"]
        ref_text_raw = prompt["ref_text"]

        if not isinstance(ref_code_raw, list):
            raise TypeError("`custom_voice_prompt.ref_code` must be a list.")
        if not isinstance(ref_text_raw, list):
            raise TypeError("`custom_voice_prompt.ref_text` must be a list.")

        ref_code: list[torch.Tensor] = []
        for item in ref_code_raw:
            if not isinstance(item, torch.Tensor):
                raise TypeError("`custom_voice_prompt.ref_code` items must be Tensor.")
            ref_code.append(item)

        ref_text: list[str] = []
        for item in ref_text_raw:
            if not isinstance(item, str):
                raise TypeError("`custom_voice_prompt.ref_text` items must be strings.")
            if item.strip() == "":
                raise ValueError(
                    "`custom_voice_prompt.ref_text` items must be non-empty."
                )
            ref_text.append(item)

        if len(ref_code) != len(ref_text):
            raise ValueError(
                "All `custom_voice_prompt` fields must have the same batch size."
            )

        return CustomVoicePromptDict(ref_code=ref_code, ref_text=ref_text)

    def _coerce_custom_voice_prompt_single(
        self, prompt: CustomVoicePromptSingleInput
    ) -> CustomVoicePromptSingleDict:
        required_keys = ("ref_code", "ref_text")
        for key in required_keys:
            if key not in prompt:
                raise KeyError(f"Missing key `{key}` in `custom_voice_prompt`.")

        ref_code_raw = prompt["ref_code"]
        if not isinstance(ref_code_raw, torch.Tensor):
            raise TypeError("`custom_voice_prompt.ref_code` must be Tensor.")

        ref_text_raw = prompt["ref_text"]
        if not isinstance(ref_text_raw, str):
            raise TypeError("`custom_voice_prompt.ref_text` must be a string.")
        if ref_text_raw.strip() == "":
            raise ValueError("`custom_voice_prompt.ref_text` must be non-empty.")

        return CustomVoicePromptSingleDict(
            ref_code=ref_code_raw,
            ref_text=ref_text_raw,
        )

    @torch.no_grad()
    def generate_custom_voice(
        self,
        tts_input: TTSInput,
        speaker: SpeakerConfiguration | torch.Tensor,
        *,
        language: str = "Auto",
        non_streaming_mode: bool = True,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
        ref_audio: AudioLike | None = None,
        ref_text: str = "",
        custom_voice_prompt: CustomVoicePromptSingleInput
        | CustomVoicePromptItem
        | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate one assistant waveform per turn in a shared CustomVoice
        context. An ICL reference prompt can be layered on top through
        `ref_audio`/`ref_text` or `custom_voice_prompt`.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice")

        chunks = self._normalize_tts_input(tts_input)
        self._validate_chunk_instruction_support(chunks)
        language_value = self._normalize_language_value(language)
        speaker_value = speaker

        self._validate_languages([language_value])
        if not isinstance(speaker_value, torch.Tensor):
            self._validate_speaker_configuration(speaker_value)

        custom_voice_prompt_single: CustomVoicePromptSingleDict | None = None
        ref_text_for_id = ""
        if custom_voice_prompt is None:
            if ref_audio is not None:
                prompt_items = self.create_custom_voice_prompt(
                    ref_audio=[ref_audio],
                    ref_text=[ref_text],
                )
                if len(prompt_items) != 1:
                    raise ValueError(
                        "Single generation requires exactly one custom voice prompt item."
                    )
                prompt_item = prompt_items[0]
                custom_voice_prompt_single = (
                    self._prompt_item_to_custom_voice_prompt_single(prompt_item)
                )
                ref_text_for_id = prompt_item.ref_text
            elif ref_text != "":
                raise ValueError(
                    "`ref_text` requires `ref_audio` or `custom_voice_prompt`."
                )
        else:
            if ref_audio is not None:
                raise ValueError(
                    "Pass either `custom_voice_prompt` or `ref_audio`/`ref_text`, not both."
                )
            if ref_text != "":
                raise ValueError(
                    "`ref_text` is already included in `custom_voice_prompt`."
                )
            if isinstance(custom_voice_prompt, CustomVoicePromptItem):
                prompt_item = custom_voice_prompt
                custom_voice_prompt_single = (
                    self._prompt_item_to_custom_voice_prompt_single(prompt_item)
                )
            else:
                custom_voice_prompt_single = self._coerce_custom_voice_prompt_single(
                    custom_voice_prompt
                )
            ref_text_for_id = custom_voice_prompt_single["ref_text"]

        input_ids, instruct_ids = self._tokenize_custom_voice_chunks(chunks)
        ref_id = self._tokenize_ref_text(ref_text_for_id)

        generation_options = self._resolve_generation_options(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        ref_code = (
            custom_voice_prompt_single["ref_code"]
            if custom_voice_prompt_single is not None
            else None
        )
        talker_codes_list, _ = self.model.generate_custom_voice_turns(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            language=language_value,
            speaker=speaker_value,
            ref_code=ref_code,
            ref_id=ref_id,
            use_icl_prompt=custom_voice_prompt_single is not None,
            non_streaming_mode=non_streaming_mode,
            **generation_options,
        )

        return self._decode_talker_turns(
            talker_codes_list,
            prefix_code=ref_code if custom_voice_prompt_single is not None else None,
        )

    @torch.no_grad()
    def generate_custom_voice_batch(
        self,
        tts_input: TTSBatchInput,
        speaker: SpeakerBatchInput,
        *,
        language: StringBatchInput = (),
        non_streaming_mode: bool = True,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
        ref_audio: AudioBatchInput | None = None,
        ref_text: StringBatchInput = (),
        custom_voice_prompt: CustomVoicePromptInput
        | list[CustomVoicePromptItem]
        | None = None,
    ) -> tuple[list[list[np.ndarray]], int]:
        """
        Generate batched shared-context turns. Batched ICL reference prompts
        can optionally be layered on top.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice_batch")

        structured_inputs = self._normalize_tts_batch_input(tts_input)
        for chunks in structured_inputs:
            self._validate_chunk_instruction_support(chunks)
        languages = self._normalize_language_values(language, len(structured_inputs))
        speakers = list(speaker)

        if not (len(structured_inputs) == len(languages) == len(speakers)):
            raise ValueError(
                "Batch size mismatch: "
                f"tts_input={len(structured_inputs)}, language={len(languages)}, "
                f"speaker={len(speakers)}"
            )

        self._validate_languages(languages)
        for speaker_value in speakers:
            if isinstance(speaker_value, torch.Tensor):
                continue
            self._validate_speaker_configuration(speaker_value)

        custom_voice_prompt_dict: CustomVoicePromptDict | None = None
        ref_texts_for_ids: list[str] = []
        if custom_voice_prompt is None:
            if ref_audio is not None:
                prompt_items = self.create_custom_voice_prompt(
                    ref_audio=list(ref_audio),
                    ref_text=ref_text,
                )
                if len(prompt_items) != len(structured_inputs):
                    raise ValueError(
                        "Batch size mismatch: "
                        f"prompt={len(prompt_items)}, tts_input={len(structured_inputs)}"
                    )
                custom_voice_prompt_dict = self._prompt_items_to_custom_voice_prompt(
                    prompt_items
                )
                ref_texts_for_ids = [item.ref_text for item in prompt_items]
            elif len(ref_text) != 0:
                raise ValueError(
                    "`ref_text` requires `ref_audio` or `custom_voice_prompt`."
                )
        else:
            if ref_audio is not None:
                raise ValueError(
                    "Pass either `custom_voice_prompt` or `ref_audio`/`ref_text`, not both."
                )
            if len(ref_text) != 0:
                raise ValueError(
                    "`ref_text` is already included in `custom_voice_prompt`."
                )
            if isinstance(custom_voice_prompt, list):
                prompt_items: list[CustomVoicePromptItem] = []
                for item in custom_voice_prompt:
                    if not isinstance(item, CustomVoicePromptItem):
                        raise TypeError(
                            "`custom_voice_prompt` list items must be CustomVoicePromptItem."
                        )
                    prompt_items.append(item)
                if len(prompt_items) != len(structured_inputs):
                    raise ValueError(
                        "Batch size mismatch: "
                        f"prompt={len(prompt_items)}, tts_input={len(structured_inputs)}"
                    )
                custom_voice_prompt_dict = self._prompt_items_to_custom_voice_prompt(
                    prompt_items
                )
                ref_texts_for_ids = [item.ref_text for item in prompt_items]
            else:
                custom_voice_prompt_dict = self._coerce_custom_voice_prompt_dict(
                    custom_voice_prompt
                )
                if len(custom_voice_prompt_dict["ref_code"]) != len(structured_inputs):
                    raise ValueError(
                        "Batch size mismatch in `custom_voice_prompt` fields and "
                        "`tts_input`."
                    )
                ref_texts_for_ids = custom_voice_prompt_dict["ref_text"]

        generation_options = self._resolve_generation_options(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        wavs_by_input: list[list[np.ndarray]] = []
        sample_rate: int | None = None
        for index, (chunks, language_value, speaker_value) in enumerate(
            zip(structured_inputs, languages, speakers)
        ):
            input_ids, instruct_ids = self._tokenize_custom_voice_chunks(chunks)
            if custom_voice_prompt_dict is None:
                ref_code = None
                ref_id = None
                use_icl_prompt = False
            else:
                ref_code = custom_voice_prompt_dict["ref_code"][index]
                ref_id = self._tokenize_ref_text(ref_texts_for_ids[index])
                use_icl_prompt = True

            talker_codes_list, _ = self.model.generate_custom_voice_turns(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                language=language_value,
                speaker=speaker_value,
                ref_code=ref_code,
                ref_id=ref_id,
                use_icl_prompt=use_icl_prompt,
                non_streaming_mode=non_streaming_mode,
                **generation_options,
            )
            wavs, fs = self._decode_talker_turns(
                talker_codes_list,
                prefix_code=ref_code if use_icl_prompt else None,
            )
            if sample_rate is not None and fs != sample_rate:
                raise RuntimeError("Decoded sample rates differ across batch items.")
            sample_rate = fs
            wavs_by_input.append(wavs)
        if sample_rate is None:
            raise RuntimeError("Structured batch generation produced no outputs.")
        return wavs_by_input, sample_rate
