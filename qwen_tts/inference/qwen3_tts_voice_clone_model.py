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
from typing import Mapping, Optional, TypedDict, Union

import librosa
import numpy as np
import torch

from .qwen3_tts_base_model import AudioLike, GenerateExtraArg, Qwen3TTSBaseModel


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """

    ref_code: Optional[torch.Tensor]  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class VoiceClonePromptDict(TypedDict):
    ref_code: list[torch.Tensor | None]
    ref_spk_embedding: list[torch.Tensor]
    x_vector_only_mode: list[bool]
    icl_mode: list[bool]


VoiceClonePromptInput = Mapping[str, Sequence[torch.Tensor | bool | None]]


class Qwen3TTSVoiceCloneModel(Qwen3TTSBaseModel):
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, list[AudioLike]],
        ref_text: Optional[Union[str, list[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, list[bool]] = False,
    ) -> list[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.
        """
        self._ensure_model_type("base", "create_voice_clone_prompt")

        speech_tokenizer = self._require_speech_tokenizer()
        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = (
            self._ensure_list(ref_text)
            if isinstance(ref_text, list)
            else ([ref_text] * len(ref_audio_list))
        )
        xvec_list = (
            self._ensure_list(x_vector_only_mode)
            if isinstance(x_vector_only_mode, list)
            else ([x_vector_only_mode] * len(ref_audio_list))
        )

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
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(
                        f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}"
                    )

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(
                    y=wav_resample.astype(np.float32),
                    orig_sr=int(sr),
                    target_sr=self.model.speaker_encoder_sample_rate,
                )

            spk_emb = self.model.extract_speaker_embedding(
                audio=wav_resample, sr=self.model.speaker_encoder_sample_rate
            )

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

    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, list[str]],
        language: Optional[Union[str, list[str]]] = None,
        ref_audio: Optional[Union[AudioLike, list[AudioLike]]] = None,
        ref_text: Optional[Union[str, list[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, list[bool]] = False,
        voice_clone_prompt: Optional[
            Union[VoiceClonePromptInput, list[VoiceClonePromptItem]]
        ] = None,
        non_streaming_mode: bool = False,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: GenerateExtraArg,
    ) -> tuple[list[np.ndarray], int]:
        """
        Voice clone speech using the Base model.
        """
        self._ensure_model_type("base", "generate_voice_clone")

        texts = self._ensure_list(text)
        if isinstance(language, list):
            languages = list(language)
        elif language is None:
            languages = ["Auto"] * len(texts)
        else:
            languages = [language] * len(texts)
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}"
            )

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError(
                    "Either `voice_clone_prompt` or `ref_audio` must be provided."
                )
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(
                    f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}"
                )
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(
                prompt_items
            )
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items_raw = list(voice_clone_prompt)
                prompt_items: list[VoiceClonePromptItem] = []
                for item in prompt_items_raw:
                    if not isinstance(item, VoiceClonePromptItem):
                        raise TypeError(
                            "`voice_clone_prompt` list items must be VoiceClonePromptItem."
                        )
                    prompt_items.append(item)
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(
                        f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}"
                    )
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(
                    prompt_items
                )
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = self._coerce_voice_clone_prompt_dict(
                    voice_clone_prompt
                )
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts_batch(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for rt in ref_texts_for_ids:
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts_batch([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        talker_codes_list, _ = self.model.generate_batch(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        codes_for_decode = []
        ref_code_list = voice_clone_prompt_dict["ref_code"]
        for i, codes in enumerate(talker_codes_list):
            ref_code_item = ref_code_list[i]
            if ref_code_item is not None:
                codes_for_decode.append(
                    torch.cat([ref_code_item.to(codes.device), codes], dim=0)
                )
            else:
                codes_for_decode.append(codes)

        speech_tokenizer = self._require_speech_tokenizer()
        wavs_all, fs = speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_for_decode]
        )

        wavs_out: list[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_item = ref_code_list[i]
            if ref_code_item is not None:
                ref_len = int(ref_code_item.shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs
