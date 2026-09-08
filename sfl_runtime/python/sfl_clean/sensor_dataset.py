"""Export labeled sensor windows for the native Llama client."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

INSTRUCTION = (
    "Given the wearable-sensor representation, output valence and arousal as "
    "integers from 1 to 5, followed by a concise assessment."
)


def make_training_record(
    *, tokenizer: Any, sensor: Iterable[float], valence: int, arousal: int,
    assessment: str, sequence_length: int,
) -> dict[str, list[int] | list[float]]:
    if valence not in range(1, 6) or arousal not in range(1, 6):
        raise ValueError("valence and arousal must be integers from 1 to 5")
    if sequence_length < 2 or not assessment.strip():
        raise ValueError("sequence_length and assessment are invalid")
    prompt_value = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Transformers 5 returns a BatchEncoding by default, while earlier
    # releases returned the token-id list directly.
    prompt = (
        list(prompt_value["input_ids"])
        if isinstance(prompt_value, Mapping)
        else list(prompt_value)
    )
    target_text = (
        f"Valence: {valence}\nArousal: {arousal}\n"
        f"Assessment: {assessment.strip()}"
    )
    target = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > sequence_length:
        available = sequence_length - len(prompt)
        if available <= 0:
            raise ValueError("sequence_length is too short for the instruction template")
        target = target[:available]
        target[-1] = tokenizer.eos_token_id
    ids = prompt + target
    padding = sequence_length - len(ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return {
        "sensor": [float(value) for value in sensor],
        "token_ids": ids + [pad_id] * padding,
        "attention_mask": [1] * len(ids) + [0] * padding,
        "loss_mask": [0] * len(prompt) + [1] * len(target) + [0] * padding,
    }


def export_jsonl(source: Path, destination: Path, model: str, sequence_length: int) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                record = make_training_record(
                    tokenizer=tokenizer,
                    sensor=value["sensor"],
                    valence=value["valence"],
                    arousal=value["arousal"],
                    assessment=value["assessment"],
                    sequence_length=sequence_length,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid source record at line {line_number}: {error}") from error
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    export_jsonl(args.input, args.output, args.model, args.sequence_length)


if __name__ == "__main__":
    main()
