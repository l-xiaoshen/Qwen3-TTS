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
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSCustomVoiceModel, TTSInput


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    tts = Qwen3TTSCustomVoiceModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # -------- Single (with instruct) --------
    torch.cuda.synchronize()
    t0 = time.time()

    single_input: TTSInput = [
        {
            "instruction": "用特别愤怒的语气说",
            "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        }
    ]
    wavs, sr = tts.generate_custom_voice(
        tts_input=single_input,
        language="Chinese",
        speaker={"Vivian": 1.0},
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # -------- Batch (some empty instruct) --------
    tts_inputs: list[TTSInput] = [
        [
            {
                "instruction": "",
                "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
            }
        ],
        [
            {
                "instruction": "Very happy.",
                "text": "She said she would be here by noon.",
            }
        ],
    ]
    languages = ["Chinese", "English"]
    speakers = [{"Vivian": 1.0}, {"Ryan": 1.0}]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs_batch, sr = tts.generate_custom_voice_batch(
        tts_inputs=tts_inputs,
        language=languages,
        speaker=speakers,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, request_wavs in enumerate(wavs_batch):
        for j, wav in enumerate(request_wavs):
            sf.write(f"qwen3_tts_test_custom_batch_{i}_{j}.wav", wav, sr)


if __name__ == "__main__":
    main()
