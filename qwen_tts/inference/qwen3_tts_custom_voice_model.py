# coding=utf-8
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
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Mapping, TypedDict

import numpy as np
import torch

from qwen_tts.core import SpeakerConfiguration, SubTalkerConfiguration

from ..core.models import Qwen3TTSCustomVoiceForConditionalGeneration
from .qwen3_tts_base_model import (
    AudioLike,
    GenerateExtraArg,
    Qwen3TTSBaseModel,
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


class Qwen3TTSCustomVoiceModel(Qwen3TTSBaseModel):
    model: Qwen3TTSCustomVoiceForConditionalGeneration

    @torch.inference_mode()
    def create_custom_voice_prompt(
        self,
        ref_audio: Sequence[AudioLike],
        ref_text: Sequence[str],
    ) -> list[CustomVoicePromptItem]:
        """
        Build reusable ICL prompt items from reference audio/text for the
        CustomVoice model. Speaker identity must still be supplied separately
        through `speaker` during generation.
        """
        self._ensure_model_type("custom_voice", "create_custom_voice_prompt")

        speech_tokenizer = self._require_speech_tokenizer()
        if not isinstance(ref_audio, Sequence) or isinstance(ref_audio, (str, bytes)):
            raise TypeError("`ref_audio` must be a sequence of audio inputs.")
        ref_audio_list: list[AudioLike] = list(ref_audio)

        if not isinstance(ref_text, Sequence) or isinstance(ref_text, (str, bytes)):
            raise TypeError("`ref_text` must be a sequence of strings.")
        ref_text_list: list[str] = []
        for item in ref_text:
            if not isinstance(item, str):
                raise TypeError("`ref_text` items must be strings.")
            if item.strip() == "":
                raise ValueError(
                    "`ref_text` items must be non-empty for CustomVoice ICL prompting."
                )
            ref_text_list.append(item)

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

    def _prepend_ref_code_for_decode(
        self, talker_codes: torch.Tensor, ref_code: torch.Tensor | None
    ) -> torch.Tensor:
        if ref_code is None:
            return talker_codes
        return torch.cat([ref_code.to(talker_codes.device), talker_codes], dim=0)

    def _trim_decoded_ref_prefix(
        self,
        wav: np.ndarray,
        ref_code: torch.Tensor | None,
        total_code_length: int,
    ) -> np.ndarray:
        if ref_code is None:
            return wav
        ref_len = int(ref_code.shape[0])
        cut = int(ref_len / max(total_code_length, 1) * wav.shape[0])
        return wav[cut:]

    def _decode_custom_voice_codes_single(
        self, talker_codes: torch.Tensor, ref_code: torch.Tensor | None
    ) -> tuple[np.ndarray, int]:
        codes_for_decode = self._prepend_ref_code_for_decode(talker_codes, ref_code)
        wavs, fs = self._decode_talker_codes_batch([codes_for_decode])
        return (
            self._trim_decoded_ref_prefix(
                wavs[0],
                ref_code=ref_code,
                total_code_length=int(codes_for_decode.shape[0]),
            ),
            fs,
        )

    def _decode_custom_voice_codes_batch(
        self,
        talker_codes_list: Sequence[torch.Tensor],
        ref_codes: Sequence[torch.Tensor | None],
    ) -> tuple[list[np.ndarray], int]:
        if len(talker_codes_list) != len(ref_codes):
            raise ValueError(
                "Batch size mismatch between generated codes and reference codes."
            )

        codes_for_decode = [
            self._prepend_ref_code_for_decode(talker_codes, ref_code)
            for talker_codes, ref_code in zip(talker_codes_list, ref_codes)
        ]
        wavs, fs = self._decode_talker_codes_batch(codes_for_decode)
        wavs_out = [
            self._trim_decoded_ref_prefix(
                wav,
                ref_code=ref_code,
                total_code_length=int(codes.shape[0]),
            )
            for wav, ref_code, codes in zip(wavs, ref_codes, codes_for_decode)
        ]
        return wavs_out, fs

    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: str,
        speaker: SpeakerConfiguration | torch.Tensor,
        language: str = "Auto",
        instruct: str = "",
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int = 2048,
        ref_audio: AudioLike | None = None,
        ref_text: str = "",
        custom_voice_prompt: CustomVoicePromptSingleInput
        | CustomVoicePromptItem
        | None = None,
        **kwargs: GenerateExtraArg,
    ) -> tuple[np.ndarray, int]:
        """
        Generate one utterance with the CustomVoice model using a speaker
        configuration or a direct speaker embedding. Optionally, an ICL
        reference prompt can be layered on top through `ref_audio`/`ref_text`
        or `custom_voice_prompt`.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice")

        if not isinstance(text, str):
            raise TypeError("`text` must be a string.")
        if not isinstance(instruct, str):
            raise TypeError("`instruct` must be a string.")
        if not isinstance(ref_text, str):
            raise TypeError("`ref_text` must be a string.")

        language_value = self._normalize_language_value(language)
        speaker_value = speaker

        model_size = self.model.tts_model_size
        if isinstance(model_size, str) and model_size in "0b6":
            # for 0b6 model, instruct is not supported
            instruct_value = ""
        else:
            instruct_value = instruct

        self._validate_languages([language_value])
        if not isinstance(speaker_value, torch.Tensor):
            if not isinstance(speaker_value, dict):
                raise TypeError(
                    "`speaker` must be a speaker configuration dictionary or a tensor embedding."
                )
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

        input_segments = self._build_runtime_segments(
            text=text,
            instruct=instruct_value,
        )
        input_role_id, input_text_id = self._tokenize_assistant_segments(input_segments)
        instruct_id = self._tokenize_instruction_segments(input_segments)
        ref_id = self._tokenize_reference_segments(
            self._build_reference_segments(ref_text_for_id)
        )

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes, _ = self.model.generate_custom_voice(
            input_role_id=input_role_id,
            input_text_id=input_text_id,
            instruct_id=instruct_id,
            language=language_value,
            speaker=speaker_value,
            ref_code=(
                custom_voice_prompt_single["ref_code"]
                if custom_voice_prompt_single is not None
                else None
            ),
            ref_id=ref_id,
            use_icl_prompt=custom_voice_prompt_single is not None,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        if custom_voice_prompt_single is not None:
            return self._decode_custom_voice_codes_single(
                talker_codes,
                custom_voice_prompt_single["ref_code"],
            )
        wavs, fs = self._decode_talker_codes_batch([talker_codes])
        return wavs[0], fs

    @torch.no_grad()
    def generate_custom_voice_batch(
        self,
        text: list[str],
        speaker: Sequence[SpeakerConfiguration | torch.Tensor],
        language: Sequence[str] = (),
        instruct: Sequence[str] = (),
        non_streaming_mode: bool = True,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int = 2048,
        ref_audio: Sequence[AudioLike] | None = None,
        ref_text: Sequence[str] = (),
        custom_voice_prompt: CustomVoicePromptInput
        | list[CustomVoicePromptItem]
        | None = None,
        **kwargs: GenerateExtraArg,
    ) -> tuple[list[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using speaker
        configurations or direct speaker embeddings. Optionally, batched ICL
        reference prompts can be layered on top.
        """
        self._ensure_model_type("custom_voice", "generate_custom_voice_batch")

        if not isinstance(text, list):
            raise TypeError("`text` must be a list of strings.")
        texts: list[str] = []
        for item in text:
            if not isinstance(item, str):
                raise TypeError("`text` list items must be strings.")
            texts.append(item)

        languages = self._normalize_language_values(language, len(texts))

        if not isinstance(speaker, Sequence) or isinstance(speaker, (str, bytes)):
            raise TypeError(
                "`speaker` must be a sequence of speaker configuration dictionaries or tensor embeddings."
            )
        speakers = list(speaker)

        model_size = self.model.tts_model_size
        if isinstance(model_size, str) and model_size in "0b6":
            # for 0b6 model, instruct is not supported
            instructs = [""] * len(texts)
        elif not isinstance(instruct, Sequence) or isinstance(instruct, (str, bytes)):
            raise TypeError("`instruct` must be a sequence of strings.")
        elif len(instruct) == 0:
            instructs = [""] * len(texts)
        else:
            instructs = []
            for item in instruct:
                if not isinstance(item, str):
                    raise TypeError("`instruct` list items must be strings.")
                instructs.append(item)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        for speaker_value in speakers:
            if isinstance(speaker_value, torch.Tensor):
                continue
            if not isinstance(speaker_value, dict):
                raise TypeError(
                    "`speaker` items must be speaker configuration dictionaries or tensor embeddings."
                )
            self._validate_speaker_configuration(speaker_value)

        custom_voice_prompt_dict: CustomVoicePromptDict | None = None
        ref_texts_for_ids: list[str] = []
        if custom_voice_prompt is None:
            if ref_audio is not None:
                if not isinstance(ref_audio, Sequence) or isinstance(
                    ref_audio, (str, bytes)
                ):
                    raise TypeError("`ref_audio` must be a sequence of audio inputs.")
                prompt_items = self.create_custom_voice_prompt(
                    ref_audio=list(ref_audio),
                    ref_text=ref_text,
                )
                if len(prompt_items) != len(texts):
                    raise ValueError(
                        f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}"
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
                prompt_items_raw = list(custom_voice_prompt)
                prompt_items: list[CustomVoicePromptItem] = []
                for item in prompt_items_raw:
                    if not isinstance(item, CustomVoicePromptItem):
                        raise TypeError(
                            "`custom_voice_prompt` list items must be CustomVoicePromptItem."
                        )
                    prompt_items.append(item)
                if len(prompt_items) != len(texts):
                    raise ValueError(
                        f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}"
                    )
                custom_voice_prompt_dict = self._prompt_items_to_custom_voice_prompt(
                    prompt_items
                )
                ref_texts_for_ids = [item.ref_text for item in prompt_items]
            else:
                custom_voice_prompt_dict = self._coerce_custom_voice_prompt_dict(
                    custom_voice_prompt
                )
                if len(custom_voice_prompt_dict["ref_code"]) != len(texts):
                    raise ValueError(
                        "Batch size mismatch in `custom_voice_prompt` fields and `text`."
                    )
                ref_texts_for_ids = custom_voice_prompt_dict["ref_text"]

        if len(ref_texts_for_ids) == 0:
            ref_texts_for_ids = [""] * len(texts)

        input_role_ids: list[torch.Tensor] = []
        input_text_ids: list[torch.Tensor] = []
        instruct_ids: list[torch.Tensor | None] = []
        ref_ids: list[torch.Tensor | None] = []
        for text_value, instruct_value, ref_text_value in zip(
            texts, instructs, ref_texts_for_ids, strict=True
        ):
            input_segments = self._build_runtime_segments(
                text=text_value,
                instruct=instruct_value,
            )
            input_role_id, input_text_id = self._tokenize_assistant_segments(
                input_segments
            )
            input_role_ids.append(input_role_id)
            input_text_ids.append(input_text_id)
            instruct_ids.append(self._tokenize_instruction_segments(input_segments))
            ref_ids.append(
                self._tokenize_reference_segments(
                    self._build_reference_segments(ref_text_value)
                )
            )

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_configuration=subtalker_configuration,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes_list, _ = self.model.generate_custom_voice_batch(
            input_role_ids=input_role_ids,
            input_text_ids=input_text_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            ref_codes=(
                custom_voice_prompt_dict["ref_code"]
                if custom_voice_prompt_dict is not None
                else ()
            ),
            ref_ids=ref_ids,
            use_icl_prompts=(
                [True] * len(texts) if custom_voice_prompt_dict is not None else ()
            ),
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        if custom_voice_prompt_dict is not None:
            return self._decode_custom_voice_codes_batch(
                talker_codes_list,
                custom_voice_prompt_dict["ref_code"],
            )
        return self._decode_talker_codes_batch(talker_codes_list)
