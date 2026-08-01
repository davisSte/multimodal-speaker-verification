#!/usr/bin/env python3
"""Create reproducible speaker-level train/validation manifest splits."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    """One row from an audio manifest."""

    speaker_id: str
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create speaker-level train/validation subsets from audio manifests."
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        required=True,
        help="Training source manifest with lines: '<speaker_id> <audio_path>'.",
    )
    parser.add_argument(
        "--val-source",
        type=Path,
        required=True,
        help="Validation source manifest with lines: '<speaker_id> <audio_path>'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for split manifests, speaker lists, and metadata.",
    )
    parser.add_argument(
        "--num-train-speakers",
        type=int,
        default=2000,
        help="Number of train speakers to sample. Use 0 to keep all train speakers.",
    )
    parser.add_argument(
        "--num-val-speakers",
        type=int,
        default=0,
        help="Number of validation speakers to sample. Use 0 to keep all validation speakers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for speaker sampling.")
    parser.add_argument(
        "--train-prefix",
        type=str,
        default="train",
        help="Filename prefix for training outputs.",
    )
    parser.add_argument(
        "--val-prefix",
        type=str,
        default="validation",
        help="Filename prefix for validation outputs.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load a whitespace-delimited audio manifest."""

    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad manifest line {line_num} in {path}: {line!r}")
            rows.append(ManifestRow(speaker_id=parts[0], raw=line))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def choose_speakers(speakers: set[str], num_requested: int, rng: random.Random) -> list[str]:
    """Choose a deterministic speaker subset."""

    ordered = sorted(speakers)
    if num_requested == 0:
        return ordered
    if num_requested < 0:
        raise ValueError(f"Requested a negative number of speakers: {num_requested}")
    if num_requested > len(ordered):
        raise ValueError(f"Requested {num_requested} speakers but found only {len(ordered)}.")
    return sorted(rng.sample(ordered, num_requested))


def filter_rows(rows: list[ManifestRow], speaker_ids: set[str]) -> list[ManifestRow]:
    return [row for row in rows if row.speaker_id in speaker_ids]


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.raw + "\n")


def write_speaker_list(path: Path, speaker_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{speaker_id}\n" for speaker_id in speaker_ids), encoding="utf-8")


def summarize(rows: list[ManifestRow]) -> dict[str, int]:
    return {
        "num_rows": len(rows),
        "num_speakers": len({row.speaker_id for row in rows}),
    }


def validate_subset(name: str, rows: list[ManifestRow], speakers: list[str]) -> None:
    found = {row.speaker_id for row in rows}
    expected = set(speakers)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"{name} speaker mismatch: missing={len(missing)} extra={len(extra)}")


def main() -> None:
    args = parse_args()

    train_source = args.train_source.expanduser().resolve()
    val_source = args.val_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not train_source.is_file():
        raise FileNotFoundError(f"Train source manifest not found: {train_source}")
    if not val_source.is_file():
        raise FileNotFoundError(f"Validation source manifest not found: {val_source}")

    rng = random.Random(args.seed)
    train_rows = load_manifest(train_source)
    val_rows = load_manifest(val_source)

    train_speakers_all = {row.speaker_id for row in train_rows}
    val_speakers_all = {row.speaker_id for row in val_rows}
    overlap = train_speakers_all & val_speakers_all
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Train and validation manifests share {len(overlap)} speakers. Examples: {examples}")

    train_speakers = choose_speakers(train_speakers_all, args.num_train_speakers, rng)
    val_speakers = choose_speakers(val_speakers_all, args.num_val_speakers, rng)
    train_remaining_speakers = sorted(train_speakers_all - set(train_speakers))
    val_remaining_speakers = sorted(val_speakers_all - set(val_speakers))

    train_subset = filter_rows(train_rows, set(train_speakers))
    val_subset = filter_rows(val_rows, set(val_speakers))
    train_remaining = filter_rows(train_rows, set(train_remaining_speakers))
    val_remaining = filter_rows(val_rows, set(val_remaining_speakers))

    validate_subset("train_subset", train_subset, train_speakers)
    validate_subset("validation_subset", val_subset, val_speakers)
    validate_subset("train_remaining", train_remaining, train_remaining_speakers)
    validate_subset("validation_remaining", val_remaining, val_remaining_speakers)

    outputs = {
        "train_manifest": output_dir / f"{args.train_prefix}_manifest.txt",
        "train_manifest_remaining": output_dir / f"{args.train_prefix}_remaining_manifest.txt",
        "validation_manifest": output_dir / f"{args.val_prefix}_manifest.txt",
        "validation_manifest_remaining": output_dir / f"{args.val_prefix}_remaining_manifest.txt",
        "train_speakers": output_dir / f"{args.train_prefix}_speakers.txt",
        "train_speakers_remaining": output_dir / f"{args.train_prefix}_remaining_speakers.txt",
        "validation_speakers": output_dir / f"{args.val_prefix}_speakers.txt",
        "validation_speakers_remaining": output_dir / f"{args.val_prefix}_remaining_speakers.txt",
        "metadata": output_dir / "split_metadata.json",
    }

    write_manifest(outputs["train_manifest"], train_subset)
    write_manifest(outputs["train_manifest_remaining"], train_remaining)
    write_manifest(outputs["validation_manifest"], val_subset)
    write_manifest(outputs["validation_manifest_remaining"], val_remaining)
    write_speaker_list(outputs["train_speakers"], train_speakers)
    write_speaker_list(outputs["train_speakers_remaining"], train_remaining_speakers)
    write_speaker_list(outputs["validation_speakers"], val_speakers)
    write_speaker_list(outputs["validation_speakers_remaining"], val_remaining_speakers)

    metadata = {
        "train_source": str(train_source),
        "validation_source": str(val_source),
        "seed": args.seed,
        "num_train_speakers_requested": args.num_train_speakers,
        "num_validation_speakers_requested": args.num_val_speakers,
        "train_source_summary": summarize(train_rows),
        "validation_source_summary": summarize(val_rows),
        "train_subset_summary": summarize(train_subset),
        "validation_subset_summary": summarize(val_subset),
        "train_remaining_summary": summarize(train_remaining),
        "validation_remaining_summary": summarize(val_remaining),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
