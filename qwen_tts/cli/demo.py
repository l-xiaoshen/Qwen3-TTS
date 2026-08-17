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
"""
A gradio demo for Qwen3 TTS models.
"""

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, cast

import gradio as gr
import numpy as np
import torch

from .. import (
    Qwen3TTSBaseModel,
    Qwen3TTSCustomVoiceModel,
    Qwen3TTSVoiceCloneModel,
    Qwen3TTSVoiceDesignModel,
    SubTalkerConfiguration,
    VoiceClonePromptItem,
)
from ..core.models import (
    Qwen3TTSCustomVoiceForConditionalGeneration,
    Qwen3TTSVoiceCloneForConditionalGeneration,
    Qwen3TTSVoiceDesignForConditionalGeneration,
)

Qwen3TTSFeatureModel = (
    Qwen3TTSCustomVoiceModel | Qwen3TTSVoiceDesignModel | Qwen3TTSVoiceCloneModel
)


class DemoGenKwargs(TypedDict, total=False):
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    subtalker_configuration: SubTalkerConfiguration


class DemoLaunchKwargs(TypedDict, total=False):
    server_name: str
    server_port: int
    share: bool
    ssl_verify: bool
    ssl_certfile: str
    ssl_keyfile: str


class DemoArgs(argparse.Namespace):
    checkpoint_pos: str | None
    checkpoint: str | None
    device: str
    dtype: str
    flash_attn: bool
    ip: str
    port: int
    share: bool
    concurrency: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    ssl_verify: bool
    max_new_tokens: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    subtalker_configuration: str | None


class SavedVoiceClonePromptItem(TypedDict):
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


class SavedVoiceClonePromptPayload(TypedDict):
    items: list[SavedVoiceClonePromptItem]


GradioAudio = tuple[int, np.ndarray]
GenerationResult = tuple[GradioAudio | None, str]
FileResult = tuple[str | None, str]
ModelKind = Literal["custom_voice", "voice_design", "base"]


def _title_case_display(s: str) -> str:
    s = s.strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(
    items: list[str] | None,
) -> tuple[list[str], dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.01\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # Positional checkpoint (also supports -c/--checkpoint)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    # Model loading / from_pretrained args
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for device_map, e.g. cpu, cuda, cuda:0 (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens for generation (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k sampling (optional)."
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p sampling (optional)."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (optional).",
    )
    parser.add_argument(
        "--subtalker-configuration",
        default=None,
        help=(
            "JSON object for subtalker generation settings, for example "
            '\'{"top_k": 32, "top_p": 0.95, "temperature": 0.8}\'.'
        ),
    )

    return parser


def _resolve_checkpoint(args: DemoArgs) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)  # main() prints help
    return ckpt


def _collect_gen_kwargs(args: DemoArgs) -> DemoGenKwargs:
    mapping: DemoGenKwargs = {}
    if args.max_new_tokens is not None:
        mapping["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature is not None:
        mapping["temperature"] = float(args.temperature)
    if args.top_k is not None:
        mapping["top_k"] = int(args.top_k)
    if args.top_p is not None:
        mapping["top_p"] = float(args.top_p)
    if args.repetition_penalty is not None:
        mapping["repetition_penalty"] = float(args.repetition_penalty)
    if args.subtalker_configuration is not None:
        try:
            raw_subtalker_configuration = json.loads(args.subtalker_configuration)
        except json.JSONDecodeError as exc:
            raise ValueError("`--subtalker-configuration` must be valid JSON.") from exc
        if not isinstance(raw_subtalker_configuration, dict):
            raise ValueError("`--subtalker-configuration` must be a JSON object.")

        subtalker_configuration: SubTalkerConfiguration = {}
        for key, value in raw_subtalker_configuration.items():
            if key == "do_sample":
                if not isinstance(value, bool):
                    raise ValueError(
                        "`subtalker_configuration.do_sample` must be a boolean."
                    )
                subtalker_configuration["do_sample"] = value
            elif key == "top_k":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        "`subtalker_configuration.top_k` must be an integer."
                    )
                subtalker_configuration["top_k"] = value
            elif key == "top_p":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("`subtalker_configuration.top_p` must be numeric.")
                subtalker_configuration["top_p"] = float(value)
            elif key == "temperature":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        "`subtalker_configuration.temperature` must be numeric."
                    )
                subtalker_configuration["temperature"] = float(value)
            else:
                raise ValueError(
                    "Unsupported `subtalker_configuration` key: "
                    f"{key!r}. Supported keys are "
                    "'do_sample', 'top_k', 'top_p', and 'temperature'."
                )
        mapping["subtalker_configuration"] = subtalker_configuration
    return mapping


def _normalize_audio(wav: object, eps: float = 1e-12, clip: bool = True) -> np.ndarray:
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _external_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        result[str(key)] = item
    return result


def _audio_to_tuple(audio: object) -> tuple[np.ndarray, int] | None:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2:
        sr = audio[0]
        wav = audio[1]
        if isinstance(sr, int):
            wav = _normalize_audio(wav)
            return wav, sr

    audio_map = _external_mapping(audio)
    if audio_map is not None and "sampling_rate" in audio_map and "data" in audio_map:
        sr = audio_map["sampling_rate"]
        if not isinstance(sr, (int, float)):
            return None
        sr = int(sr)
        wav = _normalize_audio(audio_map["data"])
        return wav, sr

    return None


def _tensor_from_external(value: object, field_name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return torch.tensor(list(value))
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise TypeError(f"{field_name} must be tensor-like.") from exc
    raise TypeError(f"{field_name} must be tensor-like.")


def _external_file_path(file_obj: object) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, os.PathLike):
        path_value = os.fspath(file_obj)
        if isinstance(path_value, str) and path_value:
            return path_value
    name: object = getattr(file_obj, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(name, os.PathLike):
        name_value = os.fspath(name)
        if isinstance(name_value, str) and name_value:
            return name_value
    path: object = getattr(file_obj, "path", None)
    if isinstance(path, str) and path:
        return path
    if isinstance(path, os.PathLike):
        path_value = os.fspath(path)
        if isinstance(path_value, str) and path_value:
            return path_value
    return str(file_obj)


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> GradioAudio:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _specialize_model(tts: Qwen3TTSBaseModel) -> Qwen3TTSFeatureModel:
    model = tts.model
    if isinstance(model, Qwen3TTSCustomVoiceForConditionalGeneration):
        return Qwen3TTSCustomVoiceModel(
            model=model,
            processor=tts.processor,
            generate_defaults=tts.generate_defaults,
        )
    if isinstance(model, Qwen3TTSVoiceDesignForConditionalGeneration):
        return Qwen3TTSVoiceDesignModel(
            model=model,
            processor=tts.processor,
            generate_defaults=tts.generate_defaults,
        )
    if isinstance(model, Qwen3TTSVoiceCloneForConditionalGeneration):
        return Qwen3TTSVoiceCloneModel(
            model=model,
            processor=tts.processor,
            generate_defaults=tts.generate_defaults,
        )
    raise TypeError(f"Unsupported Qwen-TTS model class: {type(model).__name__}")


def _detect_model_kind(tts: Qwen3TTSFeatureModel) -> ModelKind:
    if isinstance(tts, Qwen3TTSCustomVoiceModel):
        return "custom_voice"
    if isinstance(tts, Qwen3TTSVoiceDesignModel):
        return "voice_design"
    if isinstance(tts, Qwen3TTSVoiceCloneModel):
        return "base"
    raise TypeError(f"Unsupported Qwen-TTS wrapper: {type(tts).__name__}")


def build_demo(
    tts: Qwen3TTSFeatureModel, ckpt: str, gen_kwargs_default: DemoGenKwargs
) -> gr.Blocks:
    model_kind = _detect_model_kind(tts)

    supported_langs_raw = tts.model.get_supported_languages()
    supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_choices_and_map(
        [x for x in (supported_langs_raw or [])]
    )
    spk_choices_disp, spk_map = _build_choices_and_map(
        [x for x in (supported_spks_raw or [])]
    )

    def _gen_common_kwargs() -> DemoGenKwargs:
        return DemoGenKwargs(gen_kwargs_default)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Qwen3 TTS Demo
**Checkpoint:** `{ckpt}`  
**Model Type:** `{model_kind}`  
"""
        )

        if model_kind == "custom_voice":
            if not isinstance(tts, Qwen3TTSCustomVoiceModel):
                raise TypeError(
                    f"Expected Qwen3TTSCustomVoiceModel, got {type(tts).__name__}"
                )
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        placeholder="Enter text to synthesize (输入要合成的文本).",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                        spk_in = gr.Dropdown(
                            label="Speaker (说话人)",
                            choices=spk_choices_disp,
                            value="Vivian",
                            interactive=True,
                        )
                    instruct_in = gr.Textbox(
                        label="Instruction (Optional) (控制指令，可不输入)",
                        lines=2,
                        placeholder="e.g. Say it in a very angry tone (例如：用特别伤心的语气说).",
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_instruct(
                text: str, lang_disp: str, spk_disp: str, instruct: str
            ) -> GenerationResult:
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not spk_disp:
                        return None, "Speaker is required (必须选择说话人)."
                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    instruct_value = (instruct or "").strip()
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_custom_voice(
                        tts_input=[
                            {"text": text.strip(), "instruction": instruct_value}
                        ],
                        language=language,
                        speaker={speaker: 1.0} if speaker else {},
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished. (生成完成)"
                except Exception as exc:  # noqa: BLE001 - Report failures in the UI.
                    return None, f"{type(exc).__name__}: {exc}"

            btn.click(
                run_instruct,
                inputs=[text_in, lang_in, spk_in, instruct_in],
                outputs=[audio_out, err],
            )

        elif model_kind == "voice_design":
            if not isinstance(tts, Qwen3TTSVoiceDesignModel):
                raise TypeError(
                    f"Expected Qwen3TTSVoiceDesignModel, got {type(tts).__name__}"
                )
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(
                        label="Text (待合成文本)",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="Language (语种)",
                            choices=lang_choices_disp,
                            value="Auto",
                            interactive=True,
                        )
                    design_in = gr.Textbox(
                        label="Voice Design Instruction (音色描述)",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                    )
                    btn = gr.Button("Generate (生成)", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio (合成结果)", type="numpy")
                    err = gr.Textbox(label="Status (状态)", lines=2)

            def run_voice_design(
                text: str, lang_disp: str, design: str
            ) -> GenerationResult:
                try:
                    if not text or not text.strip():
                        return None, "Text is required (必须填写文本)."
                    if not design or not design.strip():
                        return (
                            None,
                            "Voice design instruction is required (必须填写音色描述).",
                        )
                    language = lang_map.get(lang_disp, "Auto")
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_voice_design(
                        tts_input=[
                            {"text": text.strip(), "instruction": design.strip()}
                        ],
                        language=language,
                        **kwargs,
                    )
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished. (生成完成)"
                except Exception as exc:  # noqa: BLE001 - Report failures in the UI.
                    return None, f"{type(exc).__name__}: {exc}"

            btn.click(
                run_voice_design,
                inputs=[text_in, lang_in, design_in],
                outputs=[audio_out, err],
            )

        else:  # voice_clone for base
            if not isinstance(tts, Qwen3TTSVoiceCloneModel):
                raise TypeError(
                    f"Expected Qwen3TTSVoiceCloneModel, got {type(tts).__name__}"
                )
            with gr.Tabs():
                with gr.Tab("Clone & Generate (克隆并合成)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(
                                label="Reference Audio (参考音频)",
                            )
                            ref_text = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )

                        with gr.Column(scale=2):
                            text_in = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            btn = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out = gr.Audio(
                                label="Output Audio (合成结果)", type="numpy"
                            )
                            err = gr.Textbox(label="Status (状态)", lines=2)

                    def run_voice_clone(
                        ref_aud: object,
                        ref_txt: str,
                        use_xvec: bool,
                        text: str,
                        lang_disp: str,
                    ) -> GenerationResult:
                        try:
                            if not text or not text.strip():
                                return (
                                    None,
                                    "Target text is required (必须填写待合成文本).",
                                )
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return (
                                    None,
                                    "Reference audio is required (必须上传参考音频).",
                                )
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            wavs, sr = tts.generate_voice_clone(
                                tts_input=[{"text": text.strip(), "instruction": ""}],
                                language=language,
                                ref_audio=at,
                                ref_text=ref_txt.strip() if ref_txt else "",
                                x_vector_only_mode=bool(use_xvec),
                                **kwargs,
                            )
                            return _wav_to_gradio_audio(
                                wavs[0], sr
                            ), "Finished. (生成完成)"
                        except Exception as exc:  # noqa: BLE001 - Report failures in the UI.
                            return None, f"{type(exc).__name__}: {exc}"

                    btn.click(
                        run_voice_clone,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in],
                        outputs=[audio_out, err],
                    )

                with gr.Tab("Save / Load Voice (保存/加载克隆音色)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Save Voice (保存音色)
Upload reference audio and text, choose use x-vector only or not, then save a reusable voice prompt file.  
(上传参考音频和参考文本，选择是否使用 use x-vector only 模式后保存为可复用的音色文件)
"""
                            )
                            ref_audio_s = gr.Audio(
                                label="Reference Audio (参考音频)", type="numpy"
                            )
                            ref_text_s = gr.Textbox(
                                label="Reference Text (参考音频文本)",
                                lines=2,
                                placeholder="Required if not set use x-vector only (不勾选use x-vector only时必填).",
                            )
                            xvec_only_s = gr.Checkbox(
                                label="Use x-vector only (仅用说话人向量，效果有限，但不用传入参考音频文本)",
                                value=False,
                            )
                            save_btn = gr.Button(
                                "Save Voice File (保存音色文件)", variant="primary"
                            )
                            prompt_file_out = gr.File(label="Voice File (音色文件)")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### Load Voice & Generate (加载音色并合成)
Upload a previously saved voice file, then synthesize new text.  
(上传已保存提示文件后，输入新文本进行合成)
"""
                            )
                            prompt_file_in = gr.File(
                                label="Upload Prompt File (上传提示文件)"
                            )
                            text_in2 = gr.Textbox(
                                label="Target Text (待合成文本)",
                                lines=4,
                                placeholder="Enter text to synthesize (输入要合成的文本).",
                            )
                            lang_in2 = gr.Dropdown(
                                label="Language (语种)",
                                choices=lang_choices_disp,
                                value="Auto",
                                interactive=True,
                            )
                            gen_btn2 = gr.Button("Generate (生成)", variant="primary")

                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(
                                label="Output Audio (合成结果)", type="numpy"
                            )
                            err2 = gr.Textbox(label="Status (状态)", lines=2)

                    def save_prompt(
                        ref_aud: object, ref_txt: str, use_xvec: bool
                    ) -> FileResult:
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return (
                                    None,
                                    "Reference audio is required (必须上传参考音频).",
                                )
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, (
                                    "Reference text is required when use x-vector only is NOT enabled.\n"
                                    "(未勾选 use x-vector only 时，必须提供参考音频文本；否则请勾选 use x-vector only，但效果会变差.)"
                                )
                            items = tts.create_voice_clone_prompt(
                                ref_audio=[at],
                                ref_text=[ref_txt.strip() if ref_txt else ""],
                                x_vector_only_mode=[bool(use_xvec)],
                            )
                            payload = SavedVoiceClonePromptPayload(
                                items=[
                                    SavedVoiceClonePromptItem(
                                        ref_code=item.ref_code,
                                        ref_spk_embedding=item.ref_spk_embedding,
                                        x_vector_only_mode=item.x_vector_only_mode,
                                        icl_mode=item.icl_mode,
                                        ref_text=item.ref_text,
                                    )
                                    for item in items
                                ]
                            )
                            fd, out_path = tempfile.mkstemp(
                                prefix="voice_clone_prompt_", suffix=".pt"
                            )
                            os.close(fd)
                            torch.save(payload, out_path)
                            return out_path, "Finished. (生成完成)"
                        except Exception as exc:  # noqa: BLE001 - Report failures in the UI.
                            return None, f"{type(exc).__name__}: {exc}"

                    def load_prompt_and_gen(
                        file_obj: object, text: str, lang_disp: str
                    ) -> GenerationResult:
                        try:
                            if file_obj is None:
                                return (
                                    None,
                                    "Voice file is required (必须上传音色文件).",
                                )
                            if not text or not text.strip():
                                return (
                                    None,
                                    "Target text is required (必须填写待合成文本).",
                                )

                            path = _external_file_path(file_obj)
                            payload_raw: object = torch.load(
                                path, map_location="cpu", weights_only=True
                            )
                            payload = _external_mapping(payload_raw)
                            if payload is None or "items" not in payload:
                                return None, "Invalid file format (文件格式不正确)."

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "Empty voice items (音色为空)."

                            items: list[VoiceClonePromptItem] = []
                            for item_raw in items_raw:
                                item = _external_mapping(item_raw)
                                if item is None:
                                    return (
                                        None,
                                        "Invalid item format in file (文件内部格式错误).",
                                    )
                                ref_code_raw = item.get("ref_code")
                                ref_code = (
                                    None
                                    if ref_code_raw is None
                                    else _tensor_from_external(ref_code_raw, "ref_code")
                                )
                                if ref_code is not None and ref_code.dim() not in (
                                    1,
                                    2,
                                ):
                                    return (
                                        None,
                                        "Invalid ref_code shape (音频码形状错误).",
                                    )

                                ref_spk_raw = item.get("ref_spk_embedding")
                                if ref_spk_raw is None:
                                    return (
                                        None,
                                        "Missing ref_spk_embedding (缺少说话人向量).",
                                    )
                                ref_spk = _tensor_from_external(
                                    ref_spk_raw, "ref_spk_embedding"
                                )
                                if ref_spk.dim() != 1:
                                    return None, (
                                        "Invalid ref_spk_embedding shape "
                                        "(说话人向量形状错误)."
                                    )

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(
                                            item.get("x_vector_only_mode", False)
                                        ),
                                        icl_mode=bool(
                                            item.get(
                                                "icl_mode",
                                                not bool(
                                                    item.get(
                                                        "x_vector_only_mode", False
                                                    )
                                                ),
                                            )
                                        ),
                                        ref_text=(
                                            str(item.get("ref_text"))
                                            if item.get("ref_text") is not None
                                            else ""
                                        ),
                                    )
                                )

                            if len(items) != 1:
                                return None, (
                                    "Voice file must contain exactly one item for this UI.\n"
                                    "(当前界面仅支持单条音色，请上传包含单个条目的音色文件。)"
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            wavs, sr = tts.generate_voice_clone(
                                tts_input=[{"text": text.strip(), "instruction": ""}],
                                language=language,
                                voice_clone_prompt=items[0],
                                **kwargs,
                            )
                            return _wav_to_gradio_audio(
                                wavs[0], sr
                            ), "Finished. (生成完成)"
                        except Exception as exc:  # noqa: BLE001 - Report failures in the UI.
                            return None, (
                                f"Failed to read or use voice file. Check file format/content.\n"
                                f"(读取或使用音色文件失败，请检查文件格式或内容)\n"
                                f"{type(exc).__name__}: {exc}"
                            )

                    save_btn.click(
                        save_prompt,
                        inputs=[ref_audio_s, ref_text_s, xvec_only_s],
                        outputs=[prompt_file_out, err2],
                    )
                    gen_btn2.click(
                        load_prompt_and_gen,
                        inputs=[prompt_file_in, text_in2, lang_in2],
                        outputs=[audio_out2, err2],
                    )

        gr.Markdown(
            """
**Disclaimer (免责声明)**  
- The audio is automatically generated/synthesized by an AI model solely to demonstrate the model’s capabilities; it may be inaccurate or inappropriate, does not represent the views of the developer/operator, and does not constitute professional advice. You are solely responsible for evaluating, using, distributing, or relying on this audio; to the maximum extent permitted by applicable law, the developer/operator disclaims liability for any direct, indirect, incidental, or consequential damages arising from the use of or inability to use the audio, except where liability cannot be excluded by law. Do not use this service to intentionally generate or replicate unlawful, harmful, defamatory, fraudulent, deepfake, or privacy/publicity/copyright/trademark‑infringing content; if a user prompts, supplies materials, or otherwise facilitates any illegal or infringing conduct, the user bears all legal consequences and the developer/operator is not responsible.
- 音频由人工智能模型自动生成/合成，仅用于体验与展示模型效果，可能存在不准确或不当之处；其内容不代表开发者/运营方立场，亦不构成任何专业建议。用户应自行评估并承担使用、传播或依赖该音频所产生的一切风险与责任；在适用法律允许的最大范围内，开发者/运营方不对因使用或无法使用本音频造成的任何直接、间接、附带或后果性损失承担责任（法律另有强制规定的除外）。严禁利用本服务故意引导生成或复制违法、有害、诽谤、欺诈、深度伪造、侵犯隐私/肖像/著作权/商标等内容；如用户通过提示词、素材或其他方式实施或促成任何违法或侵权行为，相关法律后果由用户自行承担，与开发者/运营方无关。
"""
        )

    return cast(gr.Blocks, demo)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv, namespace=DemoArgs())

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    base_tts = Qwen3TTSBaseModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    tts = _specialize_model(base_tts)

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: DemoLaunchKwargs = {
        "server_name": args.ip,
        "server_port": args.port,
        "share": args.share,
        "ssl_verify": bool(args.ssl_verify),
    }
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
