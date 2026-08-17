# Qwen TTS Fork

This fork extends `qwen-tts` with structured chunk inputs, speaker embeddings, reusable prompts, weighted speaker mixing, and hybrid custom voice generation.

The fork keeps `voice_clone` and `custom_voice` as separate APIs:

- Use `Qwen3TTSVoiceCloneModel` for reference-based voice cloning.
- Use `Qwen3TTSCustomVoiceModel` for custom voice generation and speaker merge.
- Use `Qwen3TTSVoiceDesignModel` for instruction-driven voice design.

## Features/API added in this fork

- Speaker embedding extraction.
- Separate single and batch generation APIs.
- Public `SpeakerConfiguration` support.
- Weighted speaker merge.
- Reusable prompt builders.
- Structured `TTSInput` chunks with an independent instruction per chunk.
- Hybrid custom voice generation with `speaker`, reference audio, and reference text.

Each generation method accepts `tts_input`, a non-empty sequence of `{"text": str, "instruction": str}` turns. Every item is serialized causally as a user instruction followed by an assistant text/audio response. Before a later turn is generated, its Transformer context includes the earlier instructions, assistant text prefills, generated codec tokens, and codec end markers. Single-request methods return one waveform per assistant turn; batch methods group those waveforms by logical `TTSInput`.

An empty instruction omits the user instruction block, matching the native single-turn API. Shared-context turns support both dual-track layouts: `non_streaming_mode=True` prefills each turn's complete text, while `False` consumes its text alongside generated codec frames. Each waveform is decoded with the reference and prior turn codes as acoustic context, then trimmed at the exact codec boundary.

Structured multi-turn generation is a fork-level experimental mode. The upstream checkpoints and API document independent utterances, not repeated assistant audio turns in one ChatML history. Retained text and codec context can therefore change later delivery based on dialogue semantics rather than preserve the first turn's style exactly.

CustomVoice 0.6B requires empty chunk instructions because that checkpoint does not support instruction conditioning.

VoiceDesign and CustomVoice can interpret age, gender, and timbre words as requests to alter perceived identity. For a fixed speaker, keep turn instructions limited to emotion and prosody and repeat required constraints on each turn rather than assuming an empty instruction inherits the prior style.

## Usage

### Extract a speaker embedding

Use the base model to extract a speaker embedding from reference audio.

```python
from qwen_tts import Qwen3TTSVoiceCloneModel

base = Qwen3TTSVoiceCloneModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
embedding = base.extract_speaker_embedding(ref_audio)
```

For batch extraction, use the batch API.

```python
embeddings = base.extract_speaker_embedding_batch(ref_audios)
```

### Generate custom voice

Use a direct speaker embedding or a weighted speaker configuration.

```python
from qwen_tts import Qwen3TTSCustomVoiceModel, TTSInput

tts = Qwen3TTSCustomVoiceModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
tts_input: TTSInput = [
    {"text": first_sentence, "instruction": "Calm and low voice."},
    {"text": second_sentence, "instruction": "Sound surprised."},
]
turn_wavs, sr = tts.generate_custom_voice(
    tts_input=tts_input,
    speaker=embedding,
)
```

For batch generation, use the batch API.

```python
turn_wavs_by_input, sr = tts.generate_custom_voice_batch(
    tts_input=[
        [{"text": first_text, "instruction": "Speak calmly."}],
        [{"text": second_text, "instruction": "Speak quickly."}],
    ],
    speaker=embeddings,
)
```

### Merge speakers

Use `SpeakerConfiguration` to mix built-in speakers.

```python
speaker = {"Vivian": 1.0, "Ryan": 0.3}
turn_wavs, sr = tts.generate_custom_voice(
    tts_input=[{"text": text, "instruction": ""}],
    speaker=speaker,
)
```

### Clone from reference audio and reference text

Use `voice_clone` when the flow is fully reference based.

```python
from qwen_tts import Qwen3TTSVoiceCloneModel

clone = Qwen3TTSVoiceCloneModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
prompt = clone.create_voice_clone_prompt(
    ref_audio=[ref_audio],
    ref_text=[ref_text],
)[0]
turn_wavs, sr = clone.generate_voice_clone(
    tts_input=[{"text": text, "instruction": ""}],
    voice_clone_prompt=prompt,
)
```

### Use reference prompting with instruction

Instructions are part of each structured chunk, including for VoiceClone:

```python
turn_wavs, sr = clone.generate_voice_clone(
    tts_input=[
        {
            "text": text,
            "instruction": "Speak with restrained frustration.",
        }
    ],
    voice_clone_prompt=prompt,
)
```

The hybrid `custom_voice` flow can combine the same chunk instructions with a direct speaker embedding and reference prompt:

```python
turn_wavs, sr = tts.generate_custom_voice(
    tts_input=[
        {
            "text": text,
            "instruction": "Speak with restrained frustration.",
        }
    ],
    speaker=embedding,
    ref_audio=ref_audio,
    ref_text=ref_text,
)
```

This hybrid combines a Base-model speaker embedding and ICL reference with a CustomVoice checkpoint. It is not an upstream-supported conditioning mode and can be less stable than either native path, especially across multiple turns. Prefer Base VoiceClone for reference-speaker fidelity, or a built-in CustomVoice speaker for instruction control.

For repeated requests, precompute the prompt once and reuse it.

```python
prompt = tts.create_custom_voice_prompt(
    ref_audio=[ref_audio],
    ref_text=[ref_text],
)[0]
turn_wavs, sr = tts.generate_custom_voice(
    tts_input=[
        {
            "text": text,
            "instruction": "Speak with restrained frustration.",
        }
    ],
    speaker=embedding,
    custom_voice_prompt=prompt,
)
```

### Generation controls

All single and batch generation methods expose the same closed set of keyword-only controls: `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`, `subtalker_configuration`, `max_new_tokens`, and `eos_token_id`. Omitting a sampling override, or passing `None`, uses the checkpoint's `generation_config.json` value and then the library fallback when the checkpoint does not define one. `eos_token_id=None` uses the model's codec EOS token.

```python
turn_wavs, sr = tts.generate_custom_voice(
    tts_input=[{"text": text, "instruction": ""}],
    speaker={"Vivian": 1.0},
    top_k=40,
    temperature=0.8,
    subtalker_configuration={"top_k": 20, "temperature": 0.7},
    eos_token_id=None,
)
```

The signatures do not accept arbitrary generation keywords. Misspelled or unsupported option names therefore fail immediately with Python's standard unexpected-keyword `TypeError` and are visible to static type checkers and IDE completion.

## Example files

The repository includes the following example flows:

- `examples/test_model_12hz_custom_voice.py`: Custom voice single and batch generation.
- `examples/test_model_12hz_structured_input.py`: Alternating instruction and assistant speech turns with retained text/audio context.
- `examples/test_model_12hz_voice_merge.py`: Weighted speaker merge with `SpeakerConfiguration`.
- `examples/test_model_12hz_base.py`: Voice clone single and batch generation, including prompt reuse.
- `examples/test_model_12hz_custom_voice_hybrid.py`: Hybrid custom voice generation with structured instructions and reference prompting.

## Public classes and prompt types

The fork exports the following top-level objects:

- `Qwen3TTSVoiceCloneModel`
- `Qwen3TTSCustomVoiceModel`
- `Qwen3TTSVoiceDesignModel`
- `VoiceClonePromptItem`
- `CustomVoicePromptItem`
- `SpeakerConfiguration`
- `TTSInputItem`
- `TTSInput`
- `TTSBatchInput`
