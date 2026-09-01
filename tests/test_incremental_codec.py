import unittest

import torch
from typing_extensions import override

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2_decoder import (
    Qwen3TTSTokenizerV2Decoder,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2_incremental import (
    Qwen3TTSTokenizerV2DecodeState,
)


class IncrementalCodecTest(unittest.TestCase):
    @override
    def setUp(self) -> None:
        torch.manual_seed(0)
        config = Qwen3TTSTokenizerV2DecoderConfig(
            codebook_size=16,
            hidden_size=8,
            latent_dim=8,
            codebook_dim=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=16,
            num_hidden_layers=1,
            num_quantizers=2,
            upsample_rates=(2,),
            upsampling_ratios=(2,),
            decoder_dim=8,
            sliding_window=4,
            attention_dropout=0.0,
        )
        self.decoder = Qwen3TTSTokenizerV2Decoder(config).eval()

    def codes(self, frames: int, batch_size: int = 1) -> torch.Tensor:
        return torch.randint(
            0,
            self.decoder.config.codebook_size,
            (batch_size, self.decoder.config.num_quantizers, frames),
        )

    def incremental(
        self, codes: torch.Tensor, partition: tuple[int, ...]
    ) -> tuple[torch.Tensor, list[Qwen3TTSTokenizerV2DecodeState]]:
        states: list[Qwen3TTSTokenizerV2DecodeState] = [
            self.decoder.new_decode_state() for _ in range(int(codes.shape[0]))
        ]
        outputs: list[torch.Tensor] = []
        start = 0
        for size in partition:
            outputs.append(
                torch.cat(
                    [
                        self.decoder.incremental_decode(
                            codes[row : row + 1, :, start : start + size],
                            states[row],
                        )
                        for row in range(int(codes.shape[0]))
                    ],
                    dim=0,
                )
            )
            start += size
        self.assertEqual(start, codes.shape[-1])
        return torch.cat(outputs, dim=-1), states

    def test_incremental_matches_whole_decode_for_chunk_partitions(self) -> None:
        codes = self.codes(9)
        with torch.inference_mode():
            whole = self.decoder(codes)
            for partition in (
                (9,),
                (1,) * 9,
                (1, 3, 5),
                (2, 3, 1, 3),
                (3, 4, 2),
            ):
                with self.subTest(partition=partition):
                    incremental, states = self.incremental(codes, partition)
                    self.assertEqual(incremental.shape, whole.shape)
                    torch.testing.assert_close(incremental, whole, rtol=1e-5, atol=1e-6)
                    self.assertEqual(states[0].frame_position, 9)

    def test_chunked_decode_is_exact_across_sliding_window(self) -> None:
        codes = self.codes(12, batch_size=2)
        with torch.inference_mode():
            whole = self.decoder(codes)
            chunked = self.decoder.chunked_decode(codes, chunk_size=1)
        self.assertEqual(chunked.shape, (2, 1, 12 * self.decoder.total_upsample))
        torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)

    def test_chunked_decode_crosses_legacy_300_frame_boundary(self) -> None:
        codes = self.codes(305)
        with torch.inference_mode():
            whole = self.decoder(codes)
            chunked = self.decoder.chunked_decode(codes)
        self.assertEqual(chunked.shape, whole.shape)
        torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)

    def test_prefix_and_turns_match_one_causal_sequence(self) -> None:
        prefix = self.codes(3)
        first_turn = self.codes(4)
        second_turn = self.codes(5)
        all_codes = torch.cat((prefix, first_turn, second_turn), dim=-1)

        state = self.decoder.new_decode_state()
        with torch.inference_mode():
            whole = self.decoder(all_codes)
            prefix_wav = self.decoder.incremental_decode(prefix, state)
            first_wav = self.decoder.incremental_decode(first_turn, state)
            second_wav = self.decoder.incremental_decode(second_turn, state)

        streamed = torch.cat((prefix_wav, first_wav, second_wav), dim=-1)
        torch.testing.assert_close(streamed, whole, rtol=1e-5, atol=1e-6)
        prefix_samples = prefix.shape[-1] * self.decoder.total_upsample
        torch.testing.assert_close(
            torch.cat((first_wav, second_wav), dim=-1),
            whole[..., prefix_samples:],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(state.frame_position, all_codes.shape[-1])

    def test_interleaved_streams_are_isolated(self) -> None:
        first = self.codes(7)
        second = self.codes(7)
        first_state = self.decoder.new_decode_state()
        second_state = self.decoder.new_decode_state()

        with torch.inference_mode():
            first_parts = (
                self.decoder.incremental_decode(first[..., :2], first_state),
                self.decoder.incremental_decode(first[..., 2:], first_state),
            )
            second_parts = (
                self.decoder.incremental_decode(second[..., :4], second_state),
                self.decoder.incremental_decode(second[..., 4:], second_state),
            )
            first_whole = self.decoder(first)
            second_whole = self.decoder(second)

        torch.testing.assert_close(
            torch.cat(first_parts, dim=-1), first_whole, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            torch.cat(second_parts, dim=-1), second_whole, rtol=1e-5, atol=1e-6
        )

    def test_state_memory_is_bounded_and_empty_chunk_is_noop(self) -> None:
        state = self.decoder.new_decode_state()
        empty = self.codes(0)
        with torch.inference_mode():
            empty_wav = self.decoder.incremental_decode(empty, state)
            for _ in range(12):
                self.decoder.incremental_decode(self.codes(1), state)

        self.assertEqual(empty_wav.shape, (1, 1, 0))
        self.assertEqual(state.frame_position, 12)
        self.assertIsNotNone(state.transformer_cache)
        if state.transformer_cache is None:
            self.fail("Warm state did not retain its Transformer cache.")
        cache_layer = state.transformer_cache.layers[0]
        self.assertLessEqual(cache_layer.keys.shape[-2], 3)
        self.assertTrue(state.conv_histories)
        self.assertTrue(state.transconv_overlaps)

    def test_invalid_chunk_does_not_advance_cold_state(self) -> None:
        state = self.decoder.new_decode_state()
        with self.assertRaisesRegex(TypeError, "torch.long"):
            self.decoder.incremental_decode(self.codes(2).float(), state)
        self.assertEqual(state.frame_position, 0)
        self.assertIsNone(state.transformer_cache)

        other_decoder = Qwen3TTSTokenizerV2Decoder(self.decoder.config).eval()
        with torch.inference_mode():
            self.decoder.incremental_decode(self.codes(1), state)
        with self.assertRaisesRegex(ValueError, "different decoder"):
            other_decoder.incremental_decode(self.codes(1), state)


if __name__ == "__main__":
    unittest.main()
