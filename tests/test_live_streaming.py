import inspect
import threading
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

from qwen_tts.core.models.modeling_qwen3_tts_base import (
    Qwen3TTSConditionalGenerationBase,
)
from qwen_tts.core.models.modeling_qwen3_tts_talker_model import (
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerOutputWithPast,
)
from qwen_tts.inference.qwen3_tts_base_model import (
    Qwen3TTSBaseModel,
    TTSStreamChunk,
)
from qwen_tts.inference.qwen3_tts_custom_voice_model import Qwen3TTSCustomVoiceModel
from qwen_tts.inference.qwen3_tts_voice_clone_model import Qwen3TTSVoiceCloneModel
from qwen_tts.inference.qwen3_tts_voice_design_model import Qwen3TTSVoiceDesignModel

FrameCallback = Callable[[int, torch.Tensor], None]
TurnEndCallback = Callable[[int], None]
Producer = Callable[[FrameCallback, TurnEndCallback], None]


class _FakeDecodeStream:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode_chunk(self, codes: torch.Tensor) -> tuple[np.ndarray, int]:
        self.calls.append(codes.detach().clone())
        # Make ordering observable: every codec frame becomes four samples whose
        # value is that frame's first codebook ID.
        frame_values = codes[:, 0].detach().cpu().numpy().astype(np.float32)
        return np.repeat(frame_values, 4), 24_000


class _FakeSpeechTokenizer:
    def __init__(self, *, incremental: bool = True) -> None:
        self.incremental = incremental
        self.streams: list[_FakeDecodeStream] = []

    def supports_incremental_decode(self) -> bool:
        return self.incremental

    def create_decode_stream(self) -> _FakeDecodeStream:
        stream = _FakeDecodeStream()
        self.streams.append(stream)
        return stream

    @staticmethod
    def get_decode_upsample_rate() -> int:
        return 4


def _wrapper(tokenizer: _FakeSpeechTokenizer) -> Qwen3TTSBaseModel:
    wrapper = object.__new__(Qwen3TTSBaseModel)
    wrapper.model = cast(
        Qwen3TTSConditionalGenerationBase,
        SimpleNamespace(speech_tokenizer=tokenizer),
    )
    return wrapper


def _frame(value: int, *, codebooks: int = 2) -> torch.Tensor:
    return torch.full((1, codebooks), value, dtype=torch.long)


class LiveStreamingTest(unittest.TestCase):
    def test_talker_hook_emits_only_complete_generated_codec_frames(self) -> None:
        talker = object.__new__(Qwen3TTSTalkerForConditionalGeneration)
        received: list[torch.Tensor] = []

        def callback(frame: torch.Tensor, _hidden: torch.Tensor) -> None:
            received.append(frame)

        model_kwargs: dict[str, object] = {"codec_frame_callback": callback}
        aligned_hidden = torch.ones(1, 1, 3)

        prefill = Qwen3TTSTalkerOutputWithPast(
            hidden_states=(None, None),
            past_hidden=aligned_hidden,
            generation_step=0,
        )
        model_kwargs = talker._update_model_kwargs_for_generation(
            prefill,
            model_kwargs,
        )
        self.assertEqual(received, [])

        complete_frame = torch.arange(6, dtype=torch.long).reshape(1, 6)
        generation_step = Qwen3TTSTalkerOutputWithPast(
            hidden_states=(None, complete_frame),
            generation_step=1,
        )
        model_kwargs = talker._update_model_kwargs_for_generation(
            generation_step,
            model_kwargs,
        )

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], complete_frame)
        self.assertIs(model_kwargs["codec_frame_callback"], callback)
        self.assertIs(model_kwargs["past_hidden"], generation_step.past_hidden)

    def test_first_chunk_is_yielded_before_talker_producer_completes(self) -> None:
        tokenizer = _FakeSpeechTokenizer()
        wrapper = _wrapper(tokenizer)
        release_producer = threading.Event()
        producer_completed = threading.Event()
        producer_thread_ids: list[int] = []

        def producer(on_frame: FrameCallback, on_turn_end: TurnEndCallback) -> None:
            producer_thread_ids.append(threading.get_ident())
            on_frame(0, _frame(1))
            if not release_producer.wait(timeout=5):
                raise TimeoutError("test did not release the Talker producer")
            on_frame(0, _frame(2))
            on_turn_end(0)
            producer_completed.set()

        chunks = wrapper._stream_talker_audio(
            producer,
            codec_chunk_frames=1,
        )
        try:
            first = next(chunks)
            self.assertIsInstance(first, TTSStreamChunk)
            np.testing.assert_array_equal(first.waveform, np.full(4, 1, np.float32))
            self.assertEqual(first.sample_rate, 24_000)
            self.assertEqual(first.turn_index, 0)
            self.assertFalse(producer_completed.is_set())
            self.assertEqual(len(producer_thread_ids), 1)
            self.assertNotEqual(producer_thread_ids[0], threading.get_ident())

            release_producer.set()
            remainder = list(chunks)
            self.assertEqual(len(remainder), 1)
            np.testing.assert_array_equal(
                remainder[0].waveform,
                np.full(4, 2, np.float32),
            )
            self.assertTrue(producer_completed.is_set())
        finally:
            release_producer.set()
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    def test_prefix_order_chunking_and_multi_turn_flush(self) -> None:
        tokenizer = _FakeSpeechTokenizer()
        wrapper = _wrapper(tokenizer)
        prefix = torch.cat((_frame(90), _frame(91)), dim=0)

        def producer(on_frame: FrameCallback, on_turn_end: TurnEndCallback) -> None:
            for value in (1, 2, 3):
                on_frame(0, _frame(value))
            on_turn_end(0)
            for value in (4, 5):
                on_frame(1, _frame(value))
            on_turn_end(1)

        chunks = list(
            wrapper._stream_talker_audio(
                producer,
                prefix_code=prefix,
                codec_chunk_frames=2,
            )
        )

        self.assertEqual([chunk.turn_index for chunk in chunks], [0, 0, 1])
        self.assertEqual([chunk.sample_rate for chunk in chunks], [24_000] * 3)
        np.testing.assert_array_equal(
            np.concatenate([chunk.waveform for chunk in chunks]),
            np.repeat(np.arange(1, 6, dtype=np.float32), 4),
        )

        self.assertEqual(len(tokenizer.streams), 1)
        decode_calls = tokenizer.streams[0].calls
        self.assertEqual([call.shape[0] for call in decode_calls], [2, 2, 1, 2])
        torch.testing.assert_close(decode_calls[0], prefix)
        torch.testing.assert_close(
            decode_calls[1], torch.cat((_frame(1), _frame(2)), dim=0)
        )
        torch.testing.assert_close(decode_calls[2], _frame(3))
        torch.testing.assert_close(
            decode_calls[3], torch.cat((_frame(4), _frame(5)), dim=0)
        )

    def test_interleaved_public_iterators_use_independent_codec_state(self) -> None:
        tokenizer = _FakeSpeechTokenizer()
        wrapper = _wrapper(tokenizer)

        def make_producer(values: tuple[int, ...]) -> Producer:
            def producer(on_frame: FrameCallback, on_turn_end: TurnEndCallback) -> None:
                for value in values:
                    on_frame(0, _frame(value))
                on_turn_end(0)

            return producer

        first_stream = wrapper._stream_talker_audio(
            make_producer((1, 2)), codec_chunk_frames=1
        )
        second_stream = wrapper._stream_talker_audio(
            make_producer((7, 8)), codec_chunk_frames=1
        )

        first_a = next(first_stream)
        second_a = next(second_stream)
        first_b = next(first_stream)
        second_b = next(second_stream)
        with self.assertRaises(StopIteration):
            next(first_stream)
        with self.assertRaises(StopIteration):
            next(second_stream)

        self.assertEqual(len(tokenizer.streams), 2)
        self.assertIsNot(tokenizer.streams[0], tokenizer.streams[1])
        np.testing.assert_array_equal(first_a.waveform, np.full(4, 1, np.float32))
        np.testing.assert_array_equal(first_b.waveform, np.full(4, 2, np.float32))
        np.testing.assert_array_equal(second_a.waveform, np.full(4, 7, np.float32))
        np.testing.assert_array_equal(second_b.waveform, np.full(4, 8, np.float32))
        self.assertEqual(
            [int(call[0, 0]) for call in tokenizer.streams[0].calls],
            [1, 2],
        )
        self.assertEqual(
            [int(call[0, 0]) for call in tokenizer.streams[1].calls],
            [7, 8],
        )

    def test_worker_exception_before_first_frame_is_reraised(self) -> None:
        wrapper = _wrapper(_FakeSpeechTokenizer())

        def producer(_on_frame: FrameCallback, _on_turn_end: TurnEndCallback) -> None:
            raise RuntimeError("Talker failed before output")

        chunks = wrapper._stream_talker_audio(producer, codec_chunk_frames=1)
        with self.assertRaisesRegex(RuntimeError, "Talker failed before output"):
            next(chunks)

    def test_worker_exception_after_audio_is_reraised_after_queued_chunk(self) -> None:
        wrapper = _wrapper(_FakeSpeechTokenizer())

        def producer(on_frame: FrameCallback, _on_turn_end: TurnEndCallback) -> None:
            on_frame(0, _frame(3))
            raise RuntimeError("Talker failed after output")

        chunks = wrapper._stream_talker_audio(producer, codec_chunk_frames=1)
        first = next(chunks)
        np.testing.assert_array_equal(first.waveform, np.full(4, 3, np.float32))
        with self.assertRaisesRegex(RuntimeError, "Talker failed after output"):
            next(chunks)

    def test_close_cancels_producer_even_when_output_queue_is_full(self) -> None:
        wrapper = _wrapper(_FakeSpeechTokenizer())
        producer_stopped = threading.Event()

        def producer(on_frame: FrameCallback, _on_turn_end: TurnEndCallback) -> None:
            value = 0
            try:
                while True:
                    on_frame(0, _frame(value))
                    value += 1
            finally:
                producer_stopped.set()

        chunks = wrapper._stream_talker_audio(producer, codec_chunk_frames=1)
        next(chunks)
        close = getattr(chunks, "close", None)
        self.assertIsNotNone(close)
        cast(Callable[[], None], close)()
        self.assertTrue(
            producer_stopped.wait(timeout=2),
            "closing the iterator left its producer thread running",
        )

    def test_public_stream_methods_are_siblings_of_unchanged_generate_api(self) -> None:
        method_pairs = (
            (
                Qwen3TTSCustomVoiceModel.generate_custom_voice,
                Qwen3TTSCustomVoiceModel.generate_custom_voice_stream,
            ),
            (
                Qwen3TTSVoiceCloneModel.generate_voice_clone,
                Qwen3TTSVoiceCloneModel.generate_voice_clone_stream,
            ),
            (
                Qwen3TTSVoiceDesignModel.generate_voice_design,
                Qwen3TTSVoiceDesignModel.generate_voice_design_stream,
            ),
        )
        for generate_method, stream_method in method_pairs:
            with self.subTest(method=generate_method.__qualname__):
                generate_parameters = inspect.signature(generate_method).parameters
                stream_parameters = inspect.signature(stream_method).parameters
                self.assertNotIn("codec_chunk_frames", generate_parameters)
                self.assertIn("codec_chunk_frames", stream_parameters)
                self.assertEqual(stream_parameters["codec_chunk_frames"].default, 4)
                self.assertEqual(
                    list(generate_parameters),
                    [
                        name
                        for name in stream_parameters
                        if name != "codec_chunk_frames"
                    ],
                )
                for name, generate_parameter in generate_parameters.items():
                    stream_parameter = stream_parameters[name]
                    self.assertEqual(
                        stream_parameter.kind,
                        generate_parameter.kind,
                    )
                    self.assertEqual(
                        stream_parameter.default,
                        generate_parameter.default,
                    )
                self.assertNotIn("kwargs", stream_parameters)
                self.assertFalse(
                    any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in stream_parameters.values()
                    )
                )

    def test_invalid_chunk_size_and_non_incremental_codec_fail_clearly(self) -> None:
        def producer(_on_frame: FrameCallback, _on_turn_end: TurnEndCallback) -> None:
            raise AssertionError("producer must not start for invalid stream setup")

        with self.assertRaisesRegex(ValueError, "codec_chunk_frames"):
            _wrapper(_FakeSpeechTokenizer())._stream_talker_audio(
                producer,
                codec_chunk_frames=0,
            )

        with self.assertRaisesRegex(NotImplementedError, "incremental|12 Hz"):
            _wrapper(_FakeSpeechTokenizer(incremental=False))._stream_talker_audio(
                producer,
                codec_chunk_frames=1,
            )


if __name__ == "__main__":
    unittest.main()
