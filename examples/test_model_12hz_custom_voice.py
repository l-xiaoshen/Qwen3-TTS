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

from qwen_tts import Qwen3TTSCustomVoiceModel, SpeakerConfiguration


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

    wavs, sr = tts.generate_custom_voice(
        tts_input=[
            {
                "text": "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
                "instruction": "用特别愤怒的语气说",
            }
        ],
        language="Chinese",
        speaker={"Vivian": 1.0},
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # -------- Batch (some empty instruct) --------
    texts = [
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        "She said she would be here by noon.",
    ]
    languages = ["Chinese", "English"]
    speakers: list[SpeakerConfiguration | torch.Tensor] = [
        {"Vivian": 1.0},
        {"Ryan": 1.0},
    ]
    instructs = ["", "Very happy."]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs_by_input, sr = tts.generate_custom_voice_batch(
        tts_input=[
            [{"text": text, "instruction": instruction}]
            for text, instruction in zip(texts, instructs)
        ],
        language=languages,
        speaker=speakers,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, turn_wavs in enumerate(wavs_by_input):
        sf.write(f"qwen3_tts_test_custom_batch_{i}.wav", turn_wavs[0], sr)


if __name__ == "__main__":
    main()
