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

from qwen_tts import Qwen3TTSCustomVoiceModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    tts = Qwen3TTSCustomVoiceModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    text = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
    language = "Chinese"



    speakers = []
    for ryan_weight in [i * 0.1 for i in range(10)]:
        speakers.append({"Vivian": 1, "Ryan": ryan_weight})

    texts = [text] * 10
    languages = [language] * 10


    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice_batch(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=[],
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Merge Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        ryan = round(i * 0.1, 1)
        sf.write(f"qwen3_tts_test_voice_merge_ryan_{ryan:.1f}.wav", w, sr)


if __name__ == "__main__":
    main()
