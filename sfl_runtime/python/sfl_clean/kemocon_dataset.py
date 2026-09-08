"""Convert trusted K-EmoCon window records into the native sensor JSONL format.

The input pickle is a data interchange artifact produced by the local K-EmoCon
preprocessing pipeline.  Pickle files can execute code while loading, so this
command must only be used with a file produced by a trusted collaborator.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from .sensor_dataset import make_training_record


SPLIT_SUBJECTS: dict[str, frozenset[int]] = {
    "train": frozenset((4, 5, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 26, 27, 29, 30, 31, 32)),
    "validation": frozenset((12, 13, 28)),
    "test": frozenset((1, 8, 25)),
}
_RECORD_ID = re.compile(r"^P(?P<participant>\d+)-W(?P<window>\d+)$")


def participant_from_record_id(record_id: object) -> int:
    match = _RECORD_ID.fullmatch(record_id) if isinstance(record_id, str) else None
    if match is None:
        raise ValueError(f"invalid K-EmoCon record identifier: {record_id!r}")
    return int(match.group("participant"))


def split_for_participant(participant: int) -> str:
    matches = [name for name, subjects in SPLIT_SUBJECTS.items() if participant in subjects]
    if len(matches) != 1:
        raise ValueError(f"participant P{participant} is not assigned to exactly one split")
    return matches[0]


def _score(values: Any, start: int, count: int) -> int:
    sequence = np.asarray(values, dtype=np.float32).reshape(-1)
    selected = sequence[start : start + count]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("annotation sequence does not cover the selected window")
    # K-EmoCon labels use the integer 1--5 Likert scale.  Half-up rounding is
    # explicit here so Python's banker rounding cannot silently change labels.
    return min(5, max(1, int(math.floor(float(selected.mean()) + 0.5))))


def _resample_window(
    sequence: Any,
    *,
    source_hz: int,
    offset_seconds: float,
    window_seconds: float,
    target_length: int,
) -> list[float]:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] != 6:
        raise ValueError("sensor sequence must have shape [1, 6, time] or [6, time]")
    start = round(offset_seconds * source_hz)
    length = round(window_seconds * source_hz)
    selected = values[:, start : start + length]
    if selected.shape != (6, length) or not np.isfinite(selected).all():
        raise ValueError("sensor sequence does not cover the selected window")
    old_axis = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    new_axis = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    resampled = np.stack(
        [np.interp(new_axis, old_axis, channel) for channel in selected], axis=0
    ).astype(np.float32)
    return resampled.reshape(-1).tolist()


def _assessment(valence: int, arousal: int) -> str:
    valence_text = "pleasant" if valence >= 4 else "unpleasant" if valence <= 2 else "neutral"
    arousal_text = "high" if arousal >= 4 else "low" if arousal <= 2 else "moderate"
    return f"The window shows {valence_text} valence and {arousal_text} arousal."


def convert_pickle(
    source: Path,
    destination: Path,
    model: str,
    *,
    sequence_length: int,
    source_hz: int,
    window_seconds: float,
    target_length: int,
    offset_seconds: float,
    annotation_step_seconds: float,
    max_records: int,
    split: str,
) -> int:
    from transformers import AutoTokenizer

    with source.open("rb") as stream:
        records = pickle.load(stream)  # noqa: S301 - trusted local data only
    if not isinstance(records, dict) or not records:
        raise ValueError("K-EmoCon pickle must contain a non-empty record mapping")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    label_start = int(math.floor(offset_seconds / annotation_step_seconds))
    label_count = max(1, int(math.ceil(window_seconds / annotation_step_seconds)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for record_id, value in records.items():
            participant = participant_from_record_id(record_id)
            record_split = split_for_participant(participant)
            if split != "all" and record_split != split:
                continue
            if max_records > 0 and written >= max_records:
                break
            try:
                sensor = _resample_window(
                    value["sequence"],
                    source_hz=source_hz,
                    offset_seconds=offset_seconds,
                    window_seconds=window_seconds,
                    target_length=target_length,
                )
                valence = _score(value["valence"], label_start, label_count)
                arousal = _score(value["arousal"], label_start, label_count)
                result = make_training_record(
                    tokenizer=tokenizer,
                    sensor=sensor,
                    valence=valence,
                    arousal=arousal,
                    assessment=_assessment(valence, arousal),
                    sequence_length=sequence_length,
                )
                result.update(
                    {
                        "record_id": record_id,
                        "participant_id": participant,
                        "split": record_split,
                        "valence": valence,
                        "arousal": arousal,
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid K-EmoCon record {record_id}: {error}") from error
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
            written += 1
    if written == 0:
        raise ValueError("no K-EmoCon records were written")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--source-hz", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument(
        "--target-length", type=int, default=240,
        help="30 seconds at the OpenTSLM K-EmoCon preprocessing rate (8 Hz)",
    )
    parser.add_argument("--offset-seconds", type=float, default=0.0)
    parser.add_argument("--annotation-step-seconds", type=float, default=5.0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="all",
        help="export only the selected participant-disjoint split",
    )
    args = parser.parse_args()
    count = convert_pickle(
        args.input,
        args.output,
        args.model,
        sequence_length=args.sequence_length,
        source_hz=args.source_hz,
        window_seconds=args.window_seconds,
        target_length=args.target_length,
        offset_seconds=args.offset_seconds,
        annotation_step_seconds=args.annotation_step_seconds,
        max_records=args.max_records,
        split=args.split,
    )
    print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
