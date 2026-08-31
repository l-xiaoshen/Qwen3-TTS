import unittest
import warnings
from tempfile import TemporaryDirectory

import torch

from qwen_tts.core.models.configuration_qwen3_tts import (
    CodecSpecialTokenIds,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from qwen_tts.core.models.generation.single import Qwen3TTSGenerationSingleMixin
from qwen_tts.core.models.modeling_qwen3_tts_attention import (
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerRotaryEmbedding,
)
from qwen_tts.core.models.modeling_qwen3_tts_talker_model import (
    Qwen3TTSTalkerForConditionalGeneration,
)
from qwen_tts.core.models.modeling_qwen3_tts_talker_predictor import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2_transformer import (
    Qwen3TTSTokenizerV2DecoderRotaryEmbedding,
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)


class TransformersV5MigrationTest(unittest.TestCase):
    def predictor_config(
        self,
        *,
        hidden_size: int = 12,
        head_dim: int = 6,
        num_code_groups: int = 2,
        tie_word_embeddings: bool = False,
    ) -> Qwen3TTSTalkerCodePredictorConfig:
        return Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=16,
            hidden_size=hidden_size,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=head_dim,
            max_position_embeddings=32,
            num_code_groups=num_code_groups,
            pad_token_id=0,
            tie_word_embeddings=tie_word_embeddings,
        )

    def talker_config(
        self,
        predictor_config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        hidden_size: int = 12,
        head_dim: int = 6,
        num_code_groups: int = 2,
        tie_word_embeddings: bool = False,
    ) -> Qwen3TTSTalkerConfig:
        return Qwen3TTSTalkerConfig(
            code_predictor_config=predictor_config,
            vocab_size=16,
            hidden_size=hidden_size,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=head_dim,
            max_position_embeddings=32,
            num_code_groups=num_code_groups,
            text_hidden_size=hidden_size,
            text_vocab_size=20,
            pad_token_id=0,
            tie_word_embeddings=tie_word_embeddings,
            codec_special_token_ids=CodecSpecialTokenIds(
                bos=1,
                eos=15,
                pad=0,
                think=2,
                no_think=3,
                think_bos=4,
                think_eos=5,
            ),
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [1, 1, 1],
                "interleaved": True,
            },
        )

    def test_default_rotary_embeddings_construct(self) -> None:
        predictor_config = self.predictor_config()
        talker_config = self.talker_config(predictor_config)
        tokenizer_config = Qwen3TTSTokenizerV2DecoderConfig(
            hidden_size=12,
            num_attention_heads=2,
        )

        self.assertEqual(Qwen3TTSRotaryEmbedding(predictor_config).inv_freq.shape, (3,))
        self.assertEqual(
            Qwen3TTSTalkerRotaryEmbedding(talker_config).inv_freq.shape, (3,)
        )
        self.assertEqual(
            Qwen3TTSTokenizerV2DecoderRotaryEmbedding(tokenizer_config).inv_freq.shape,
            (3,),
        )

    def test_talker_rope_buffers_survive_pretrained_loading(self) -> None:
        predictor_config = self.predictor_config()
        model = Qwen3TTSTalkerForConditionalGeneration(
            self.talker_config(predictor_config)
        )

        with TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            reloaded = Qwen3TTSTalkerForConditionalGeneration.from_pretrained(
                checkpoint_dir
            )

        rotaries = (
            reloaded.model.rotary_emb,
            reloaded.code_predictor.model.rotary_emb,
        )
        for rotary in rotaries:
            expected, _ = rotary.rope_init_fn(rotary.config, rotary.inv_freq.device)
            torch.testing.assert_close(rotary.inv_freq, expected)
            torch.testing.assert_close(rotary.original_inv_freq, expected)

    def test_tokenizer_rope_buffers_survive_pretrained_loading(self) -> None:
        config = Qwen3TTSTokenizerV2DecoderConfig(
            hidden_size=12,
            latent_dim=12,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=24,
            num_hidden_layers=1,
            sliding_window=8,
        )
        model = Qwen3TTSTokenizerV2DecoderTransformerModel(config)

        with TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            reloaded = Qwen3TTSTokenizerV2DecoderTransformerModel.from_pretrained(
                checkpoint_dir
            )

        rotary = reloaded.rotary_emb
        expected, _ = rotary.rope_init_fn(rotary.config, rotary.inv_freq.device)
        torch.testing.assert_close(rotary.inv_freq, expected)
        torch.testing.assert_close(rotary.original_inv_freq, expected)

    def test_talker_generates_multiple_cached_steps(self) -> None:
        predictor_config = self.predictor_config()
        model = Qwen3TTSTalkerForConditionalGeneration(
            self.talker_config(predictor_config)
        ).eval()

        with torch.no_grad():
            result = model.generate(
                inputs_embeds=torch.randn(1, 2, 12),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                trailing_text_hidden=torch.randn(1, 4, 12),
                tts_pad_embed=torch.randn(1, 1, 12),
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                eos_token_id=15,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        self.assertEqual(tuple(result.sequences.shape), (1, 2))
        self.assertEqual(len(result.hidden_states), 2)

    def test_talker_repetition_penalty_with_embedded_prompt_does_not_warn(
        self,
    ) -> None:
        predictor_config = self.predictor_config()
        model = Qwen3TTSTalkerForConditionalGeneration(
            self.talker_config(predictor_config)
        ).eval()
        generation = Qwen3TTSGenerationSingleMixin()
        generation.talker = model

        with warnings.catch_warnings(), torch.no_grad():
            warnings.filterwarnings(
                "error",
                message=(r"Passing `repetition_penalty` with `inputs_embeds`.*"),
                category=UserWarning,
            )
            talker_codes, talker_hidden_states, _ = generation._run_talker_generation(
                talker_input_embeds=[torch.randn(1, 2, 12)],
                trailing_text_hidden=torch.randn(1, 4, 12),
                tts_pad_embed=torch.randn(1, 1, 12),
                suppress_tokens=[15],
                max_new_tokens=2,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                subtalker_configuration=None,
                eos_token_id=15,
                repetition_penalty=1.05,
            )

        self.assertEqual(tuple(talker_codes.shape), (1, 2))
        self.assertEqual(tuple(talker_hidden_states.shape), (1, 12))

    def test_v5_parallel_plans_and_tied_weights(self) -> None:
        predictor_config = self.predictor_config(
            num_code_groups=3,
            tie_word_embeddings=True,
        )
        model = Qwen3TTSTalkerForConditionalGeneration(
            self.talker_config(
                predictor_config,
                num_code_groups=3,
                tie_word_embeddings=True,
            )
        )

        tp_plan = model._tp_plan
        pp_plan = model._pp_plan
        self.assertIsNotNone(tp_plan)
        self.assertIsNotNone(pp_plan)
        if tp_plan is None or pp_plan is None:
            self.fail("Transformers did not initialize the parallel plans.")
        self.assertEqual(tp_plan["codec_head"], "colwise_gather_output")
        self.assertEqual(tp_plan["model.codec_embedding"], "embedding_rowwise")
        talker_base_tp_plan = model.config.base_model_tp_plan
        predictor_base_tp_plan = model.code_predictor.config.base_model_tp_plan
        self.assertIsNotNone(talker_base_tp_plan)
        self.assertIsNotNone(predictor_base_tp_plan)
        if talker_base_tp_plan is None or predictor_base_tp_plan is None:
            self.fail("Transformers did not initialize the base-model TP plans.")
        self.assertNotIn("embed_tokens", talker_base_tp_plan)
        self.assertNotIn("embed_tokens", predictor_base_tp_plan)
        predictor_tp_plan = model.code_predictor._tp_plan
        self.assertIsNotNone(predictor_tp_plan)
        if predictor_tp_plan is None:
            self.fail("Transformers did not initialize the predictor TP plan.")
        self.assertEqual(
            predictor_tp_plan["model.codec_embedding.*"],
            "embedding_rowwise",
        )
        self.assertEqual(pp_plan["codec_head"], (["hidden_states"], ["logits"]))
        self.assertIs(model.codec_head.weight, model.model.codec_embedding.weight)
        for head, embedding in zip(
            model.code_predictor.lm_head,
            model.code_predictor.model.codec_embedding,
        ):
            self.assertIs(head.weight, embedding.weight)

    def test_incompatible_predictor_tying_is_rejected(self) -> None:
        predictor_config = self.predictor_config(
            hidden_size=8,
            head_dim=4,
            tie_word_embeddings=True,
        )
        talker_config = self.talker_config(predictor_config)

        with self.assertRaisesRegex(ValueError, "matching predictor and talker"):
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
                predictor_config,
                talker_config,
            )


if __name__ == "__main__":
    unittest.main()
