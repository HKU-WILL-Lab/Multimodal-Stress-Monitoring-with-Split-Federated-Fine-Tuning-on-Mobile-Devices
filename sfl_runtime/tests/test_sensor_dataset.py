from sfl_clean.sensor_dataset import make_training_record


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages and tokenize and add_generation_prompt
        return [10, 11, 12]

    def encode(self, text, *, add_special_tokens):
        assert "Valence: 4" in text and not add_special_tokens
        return [20, 21, 22, 23]


def test_record_masks_prompt_and_padding_from_language_loss() -> None:
    record = make_training_record(
        tokenizer=FakeTokenizer(), sensor=[0.5, 1.5], valence=4, arousal=2,
        assessment="Calm trend.", sequence_length=10,
    )
    assert record["token_ids"] == [10, 11, 12, 20, 21, 22, 23, 99, 0, 0]
    assert record["attention_mask"] == [1] * 8 + [0, 0]
    assert record["loss_mask"] == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
