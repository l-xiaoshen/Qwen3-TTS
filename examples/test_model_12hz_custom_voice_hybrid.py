#!/usr/bin/env python3
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
import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSCustomVoiceModel, Qwen3TTSVoiceCloneModel


def main():
    device = "cuda:0"
    base_model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    custom_voice_model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    ref_audio = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    )
    ref_text = (
        "Okay. Yeah. I resent you. I love you. I respect you. "
        "But you know what? You blew it! And thanks to you."
    )
    target_text = (
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    )

    extractor = Qwen3TTSVoiceCloneModel.from_pretrained(
        base_model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    speaker_embedding = extractor.extract_speaker_embedding(ref_audio)
    del extractor

    tts = Qwen3TTSCustomVoiceModel.from_pretrained(
        custom_voice_model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Direct path: pass speaker embedding plus ref audio/text in one call.
    torch.cuda.synchronize()
    t0 = time.time()
    wav, sr = tts.generate_custom_voice(
        text=target_text,
        language="Auto",
        speaker=speaker_embedding,
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct="Speak with restrained frustration.",
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[Hybrid CustomVoice Direct] time: {t1 - t0:.3f}s")
    sf.write("qwen3_tts_test_custom_voice_hybrid_direct.wav", wav, sr)

    # Reuse path: precompute the ICL prompt once, then synthesize again.
    prompt = tts.create_custom_voice_prompt(
        ref_audio=[ref_audio],
        ref_text=[ref_text],
    )[0]

    torch.cuda.synchronize()
    t0 = time.time()
    wav, sr = tts.generate_custom_voice(
        text=target_text,
        language="Auto",
        speaker=speaker_embedding,
        custom_voice_prompt=prompt,
        instruct="Speak with restrained frustration.",
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[Hybrid CustomVoice Prompt Reuse] time: {t1 - t0:.3f}s")
    sf.write("qwen3_tts_test_custom_voice_hybrid_prompt.wav", wav, sr)


if __name__ == "__main__":
    main()
