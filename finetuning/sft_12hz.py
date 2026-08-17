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
import argparse
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

import torch
from accelerate.accelerator import Accelerator
from dataset import TTSBatch, TTSDataset, parse_prepared_tts_json_row
from safetensors.torch import save_file
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers.models.auto.configuration_auto import AutoConfig

from qwen_tts import Qwen3TTSBaseModel
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSVoiceCloneForConditionalGeneration,
)

target_speaker_embedding: torch.Tensor | None = None


class TrainingArgs(argparse.Namespace):
    init_model_path: str
    output_model_path: str
    train_jsonl: str
    batch_size: int
    lr: float
    num_epochs: int
    speaker_name: str


@runtime_checkable
class SpeakerEncoder(Protocol):
    def __call__(self, ref_mels: torch.Tensor) -> torch.Tensor: ...


def _json_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object.")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings.")
        result[key] = item
    return result


def train(argv: Sequence[str] | None = None) -> None:
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args(argv, namespace=TrainingArgs())

    accelerator = Accelerator(
        gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard"
    )

    model_path = args.init_model_path

    qwen3tts = Qwen3TTSBaseModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if not isinstance(qwen3tts.model, Qwen3TTSVoiceCloneForConditionalGeneration):
        raise TypeError("Fine-tuning requires a Qwen3 TTS Base model.")
    if qwen3tts.model.speaker_encoder is None:
        raise RuntimeError("The Base model speaker encoder is not initialized.")

    config_loader = cast(Callable[..., object], AutoConfig.from_pretrained)
    config_raw = config_loader(model_path)
    if not isinstance(config_raw, Qwen3TTSConfig):
        raise TypeError("AutoConfig did not return a Qwen3TTSConfig.")
    config = config_raw

    with open(args.train_jsonl) as train_file:
        train_lines = train_file.readlines()
    train_data = []
    for line_number, line in enumerate(train_lines, start=1):
        value: object = json.loads(line)
        train_data.append(
            parse_prepared_tts_json_row(value, f"training JSONL line {line_number}")
        )
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader_raw = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    optimizer_raw = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = cast(
        tuple[
            Qwen3TTSVoiceCloneForConditionalGeneration,
            Optimizer,
            DataLoader[TTSBatch],
        ],
        accelerator.prepare(qwen3tts.model, optimizer_raw, train_dataloader_raw),
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_encoder: object = getattr(model, "speaker_encoder", None)
                if speaker_encoder is None:
                    raise RuntimeError("The prepared model has no speaker encoder.")
                if not isinstance(speaker_encoder, SpeakerEncoder):
                    raise TypeError("The prepared speaker encoder is incompatible.")
                speaker_embedding_raw: object = speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                )
                if not isinstance(speaker_embedding_raw, torch.Tensor):
                    raise TypeError("Speaker encoder output must be a tensor.")
                speaker_embedding = speaker_embedding_raw.detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids)
                    * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = (
                        model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                if outputs.hidden_states is None:
                    raise RuntimeError("Talker output did not include hidden states.")
                hidden_state_layers = outputs.hidden_states[0]
                if hidden_state_layers is None or len(hidden_state_layers) == 0:
                    raise RuntimeError("Talker output hidden states are empty.")
                hidden_states = hidden_state_layers[-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _sub_talker_logits, sub_talker_loss = (
                    model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )
                )
                if sub_talker_loss is None:
                    raise RuntimeError(
                        "Sub-talker output did not include a training loss."
                    )

                if outputs.loss is None:
                    raise RuntimeError("Talker output did not include a training loss.")
                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        if accelerator.is_main_process:
            output_dir = os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            )
            shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(model_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_value: object = json.load(f)
            config_dict = _json_object(config_value, "model config")
            config_dict["tts_model_type"] = "custom_voice"
            talker_config_value = config_dict.get("talker_config", {})
            talker_config = _json_object(talker_config_value, "talker config")
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = cast(
                Qwen3TTSVoiceCloneForConditionalGeneration,
                accelerator.unwrap_model(model),
            )
            state_dict = {
                k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()
            }

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            if target_speaker_embedding is None:
                raise RuntimeError("No target speaker embedding was captured.")
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()
