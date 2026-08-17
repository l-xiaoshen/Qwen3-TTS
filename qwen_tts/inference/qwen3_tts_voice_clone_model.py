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
from typing import TypedDict

import librosa
import numpy as np
import torch

from ..core.models import (
    Qwen3TTSVoiceCloneForConditionalGeneration,
    SubTalkerConfiguration,
)
from .qwen3_tts_base_model import (
    AudioLike,
    Qwen3TTSBaseModel,
    TTSBatchInput,
    TTSInput,
)


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSVoiceCloneForConditionalGeneration.generate_voice_clone(...)`.
    """

    ref_code: torch.Tensor | None  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str = ""


class VoiceClonePromptDict(TypedDict):
    ref_code: list[torch.Tensor | None]
    ref_spk_embedding: list[torch.Tensor]
    x_vector_only_mode: list[bool]
    icl_mode: list[bool]


class VoiceClonePromptSingleDict(TypedDict):
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool


VoiceClonePromptInput = Mapping[str, Sequence[torch.Tensor | bool | None]]
VoiceClonePromptSingleInput = Mapping[str, torch.Tensor | bool | None]
AudioBatchInput = list[AudioLike] | tuple[AudioLike, ...]
StringBatchInput = list[str] | tuple[str, ...]
BoolBatchInput = list[bool] | tuple[bool, ...]


class Qwen3TTSVoiceCloneModel(Qwen3TTSBaseModel):
    model: Qwen3TTSVoiceCloneForConditionalGeneration

    def _extract_ref_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        wav_resample = wav
        if sr != self.model.speaker_encoder_sample_rate:
            wav_resample = librosa.resample(
                y=wav_resample.astype(np.float32),
                orig_sr=int(sr),
                target_sr=self.model.speaker_encoder_sample_rate,
            )
        return self.model.extract_speaker_embedding(
            audio=wav_resample, sr=self.model.speaker_encoder_sample_rate
        )

    @torch.inference_mode()
    def extract_speaker_embedding(self, ref_audio: AudioLike) -> torch.Tensor:
        """
        Extract one speaker embedding from reference audio.

        The returned tensor can be passed directly as `speaker` to
        `Qwen3TTSCustomVoiceModel.generate_custom_voice(...)`.
        """
        self._ensure_model_type("base", "extract_speaker_embedding")

        normalized = self._normalize_audio_inputs([ref_audio])
        return self._extract_ref_speaker_embedding(*normalized[0])

    @torch.inference_mode()
    def extract_speaker_embedding_batch(
        self, ref_audio: AudioBatchInput
    ) -> list[torch.Tensor]:
        """
        Extract speaker embeddings from reference audio batch.

        The returned tensors can be passed directly as `speaker` entries to
        `Qwen3TTSCustomVoiceModel.generate_custom_voice_batch(...)`.
        """
        self._ensure_model_type("base", "extract_speaker_embedding_batch")
        ref_audio_list = list(ref_audio)

        normalized = self._normalize_audio_inputs(ref_audio_list)
        return [self._extract_ref_speaker_embedding(wav, sr) for wav, sr in normalized]

    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: AudioBatchInput,
        ref_text: StringBatchInput = (),
        x_vector_only_mode: BoolBatchInput = (),
    ) -> list[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.
        """
        self._ensure_model_type("base", "create_voice_clone_prompt")

        speech_tokenizer = self._require_speech_tokenizer()
        ref_audio_list = list(ref_audio)

        if len(ref_text) == 0:
            ref_text_list: list[str] = [""] * len(ref_audio_list)
        else:
            ref_text_list = list(ref_text)

        if len(x_vector_only_mode) == 0:
            xvec_list: list[bool] = [False] * len(ref_audio_list)
        else:
            xvec_list = list(x_vector_only_mode)

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(
            ref_audio_list
        ):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
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

        items: list[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(
            zip(normalized, ref_codes, ref_text_list, xvec_list)
        ):
            if not xvec_only and rtext == "":
                raise ValueError(
                    f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}"
                )

            spk_emb = self._extract_ref_speaker_embedding(wav=wav, sr=sr)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(
        self, items: list[VoiceClonePromptItem]
    ) -> VoiceClonePromptDict:
        return VoiceClonePromptDict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    def _prompt_item_to_voice_clone_prompt_single(
        self, item: VoiceClonePromptItem
    ) -> VoiceClonePromptSingleDict:
        return VoiceClonePromptSingleDict(
            ref_code=item.ref_code,
            ref_spk_embedding=item.ref_spk_embedding,
            x_vector_only_mode=item.x_vector_only_mode,
            icl_mode=item.icl_mode,
        )

    def _coerce_voice_clone_prompt_dict(
        self, prompt: VoiceClonePromptInput
    ) -> VoiceClonePromptDict:
        required_keys = (
            "ref_code",
            "ref_spk_embedding",
            "x_vector_only_mode",
            "icl_mode",
        )
        for key in required_keys:
            if key not in prompt:
                raise KeyError(f"Missing key `{key}` in `voice_clone_prompt`.")

        ref_code_raw = prompt["ref_code"]
        ref_spk_raw = prompt["ref_spk_embedding"]
        xvec_raw = prompt["x_vector_only_mode"]
        icl_raw = prompt["icl_mode"]

        if not isinstance(ref_code_raw, list):
            raise TypeError("`voice_clone_prompt.ref_code` must be a list.")
        if not isinstance(ref_spk_raw, list):
            raise TypeError("`voice_clone_prompt.ref_spk_embedding` must be a list.")
        if not isinstance(xvec_raw, list):
            raise TypeError("`voice_clone_prompt.x_vector_only_mode` must be a list.")
        if not isinstance(icl_raw, list):
            raise TypeError("`voice_clone_prompt.icl_mode` must be a list.")

        ref_code: list[torch.Tensor | None] = []
        for item in ref_code_raw:
            if item is None:
                ref_code.append(None)
            elif isinstance(item, torch.Tensor):
                ref_code.append(item)
            else:
                raise TypeError(
                    "`voice_clone_prompt.ref_code` items must be Tensor or None."
                )

        ref_spk_embedding: list[torch.Tensor] = []
        for item in ref_spk_raw:
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    "`voice_clone_prompt.ref_spk_embedding` items must be Tensor."
                )
            ref_spk_embedding.append(item)

        x_vector_only_mode = [bool(item) for item in xvec_raw]
        icl_mode = [bool(item) for item in icl_raw]

        if not (
            len(ref_code)
            == len(ref_spk_embedding)
            == len(x_vector_only_mode)
            == len(icl_mode)
        ):
            raise ValueError(
                "All `voice_clone_prompt` fields must have the same batch size."
            )

        return VoiceClonePromptDict(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
        )

    def _coerce_voice_clone_prompt_single(
        self, prompt: VoiceClonePromptSingleInput
    ) -> VoiceClonePromptSingleDict:
        required_keys = (
            "ref_code",
            "ref_spk_embedding",
            "x_vector_only_mode",
            "icl_mode",
        )
        for key in required_keys:
            if key not in prompt:
                raise KeyError(f"Missing key `{key}` in `voice_clone_prompt`.")

        ref_code_raw = prompt["ref_code"]
        if ref_code_raw is None:
            ref_code: torch.Tensor | None = None
        elif isinstance(ref_code_raw, torch.Tensor):
            ref_code = ref_code_raw
        else:
            raise TypeError("`voice_clone_prompt.ref_code` must be Tensor or None.")

        ref_spk_raw = prompt["ref_spk_embedding"]
        if not isinstance(ref_spk_raw, torch.Tensor):
            raise TypeError("`voice_clone_prompt.ref_spk_embedding` must be Tensor.")

        return VoiceClonePromptSingleDict(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_raw,
            x_vector_only_mode=bool(prompt["x_vector_only_mode"]),
            icl_mode=bool(prompt["icl_mode"]),
        )

    @torch.no_grad()
    def generate_voice_clone(
        self,
        tts_input: TTSInput,
        *,
        language: str = "Auto",
        ref_audio: AudioLike | None = None,
        ref_text: str = "",
        x_vector_only_mode: bool = False,
        voice_clone_prompt: VoiceClonePromptSingleInput
        | VoiceClonePromptItem
        | None = None,
        non_streaming_mode: bool = False,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """
        Voice clone one assistant waveform per turn in a shared Base-model context.
        """
        self._ensure_model_type("base", "generate_voice_clone")

        chunks = self._normalize_tts_input(tts_input)
        language_value = self._normalize_language_value(language)
        self._validate_languages([language_value])

        voice_clone_prompt_single: VoiceClonePromptSingleDict
        ref_text_for_id: str
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError(
                    "Either `voice_clone_prompt` or `ref_audio` must be provided."
                )
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=[ref_audio],
                ref_text=[ref_text],
                x_vector_only_mode=[x_vector_only_mode],
            )
            if len(prompt_items) != 1:
                raise ValueError(
                    "Single generation requires exactly one voice clone prompt item."
                )
            prompt_item = prompt_items[0]
            voice_clone_prompt_single = self._prompt_item_to_voice_clone_prompt_single(
                prompt_item
            )
            ref_text_for_id = prompt_item.ref_text
        else:
            if isinstance(voice_clone_prompt, VoiceClonePromptItem):
                prompt_item = voice_clone_prompt
                voice_clone_prompt_single = (
                    self._prompt_item_to_voice_clone_prompt_single(prompt_item)
                )
                ref_text_for_id = prompt_item.ref_text
            else:
                voice_clone_prompt_single = self._coerce_voice_clone_prompt_single(
                    voice_clone_prompt
                )
                ref_text_for_id = ref_text

        if voice_clone_prompt_single["icl_mode"] and ref_text_for_id.strip() == "":
            raise ValueError("`ref_text` is required for ICL voice clone prompts.")

        input_ids, instruct_ids = self._tokenize_tts_chunks(chunks)
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

        talker_codes_list, _ = self.model.generate_voice_clone_turns(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            ref_id=ref_id,
            voice_clone_prompt=voice_clone_prompt_single,
            language=language_value,
            non_streaming_mode=non_streaming_mode,
            **generation_options,
        )

        return self._decode_talker_turns(
            talker_codes_list,
            prefix_code=(
                voice_clone_prompt_single["ref_code"]
                if voice_clone_prompt_single["icl_mode"]
                else None
            ),
        )

    @torch.no_grad()
    def generate_voice_clone_batch(
        self,
        tts_input: TTSBatchInput,
        *,
        language: StringBatchInput = (),
        ref_audio: AudioBatchInput | None = None,
        ref_text: StringBatchInput = (),
        x_vector_only_mode: BoolBatchInput = (),
        voice_clone_prompt: VoiceClonePromptInput
        | list[VoiceClonePromptItem]
        | None = None,
        non_streaming_mode: bool = False,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        subtalker_configuration: SubTalkerConfiguration | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> tuple[list[list[np.ndarray]], int]:
        """
        Voice clone batched shared-context turns.
        """
        self._ensure_model_type("base", "generate_voice_clone_batch")

        structured_inputs = self._normalize_tts_batch_input(tts_input)
        languages = self._normalize_language_values(language, len(structured_inputs))
        if len(structured_inputs) != len(languages):
            raise ValueError(
                "Batch size mismatch: "
                f"tts_input={len(structured_inputs)}, language={len(languages)}"
            )

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError(
                    "Either `voice_clone_prompt` or `ref_audio` must be provided."
                )
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=list(ref_audio),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            if len(prompt_items) != len(structured_inputs):
                raise ValueError(
                    "Batch size mismatch: "
                    f"prompt={len(prompt_items)}, tts_input={len(structured_inputs)}"
                )
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(
                prompt_items
            )
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items: list[VoiceClonePromptItem] = []
                for item in voice_clone_prompt:
                    if not isinstance(item, VoiceClonePromptItem):
                        raise TypeError(
                            "`voice_clone_prompt` list items must be VoiceClonePromptItem."
                        )
                    prompt_items.append(item)
                if len(prompt_items) != len(structured_inputs):
                    raise ValueError(
                        "Batch size mismatch: "
                        f"prompt={len(prompt_items)}, tts_input={len(structured_inputs)}"
                    )
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(
                    prompt_items
                )
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = self._coerce_voice_clone_prompt_dict(
                    voice_clone_prompt
                )
                prompt_count = len(voice_clone_prompt_dict["ref_code"])
                if prompt_count != len(structured_inputs):
                    raise ValueError(
                        "Batch size mismatch: "
                        f"prompt={prompt_count}, tts_input={len(structured_inputs)}"
                    )
                ref_texts_for_ids = (
                    [""] * prompt_count if len(ref_text) == 0 else list(ref_text)
                )
                if len(ref_texts_for_ids) != prompt_count:
                    raise ValueError(
                        "Batch size mismatch: "
                        f"prompt={prompt_count}, ref_text={len(ref_texts_for_ids)}"
                    )

        for index, (icl_mode, ref_text_value) in enumerate(
            zip(voice_clone_prompt_dict["icl_mode"], ref_texts_for_ids)
        ):
            if icl_mode and ref_text_value.strip() == "":
                raise ValueError(
                    "`ref_text` is required for ICL voice clone prompts. "
                    f"Bad index={index}"
                )

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
        for index, (chunks, language_value) in enumerate(
            zip(structured_inputs, languages)
        ):
            input_ids, instruct_ids = self._tokenize_tts_chunks(chunks)
            prompt = VoiceClonePromptSingleDict(
                ref_code=voice_clone_prompt_dict["ref_code"][index],
                ref_spk_embedding=voice_clone_prompt_dict["ref_spk_embedding"][index],
                x_vector_only_mode=voice_clone_prompt_dict["x_vector_only_mode"][index],
                icl_mode=voice_clone_prompt_dict["icl_mode"][index],
            )
            ref_id = (
                self._tokenize_ref_text(ref_texts_for_ids[index])
                if len(ref_texts_for_ids) != 0
                else None
            )
            talker_codes_list, _ = self.model.generate_voice_clone_turns(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                ref_id=ref_id,
                voice_clone_prompt=prompt,
                language=language_value,
                non_streaming_mode=non_streaming_mode,
                **generation_options,
            )
            wavs, fs = self._decode_talker_turns(
                talker_codes_list,
                prefix_code=prompt["ref_code"] if prompt["icl_mode"] else None,
            )
            if sample_rate is not None and fs != sample_rate:
                raise RuntimeError("Decoded sample rates differ across batch items.")
            sample_rate = fs
            wavs_by_input.append(wavs)
        if sample_rate is None:
            raise RuntimeError("Structured batch generation produced no outputs.")
        return wavs_by_input, sample_rate
