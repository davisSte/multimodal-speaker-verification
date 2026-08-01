#!/usr/bin/env python3
"""Build LUAR-style train/validation JSONL files from audio speaker splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LUAR train/validation JSONL files using speaker lists from an audio split."
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        required=True,
        help="Training audio-text manifest.",
    )
    parser.add_argument(
        "--val-source",
        type=Path,
        required=True,
        help="Validation audio-text manifest.",
    )
    parser.add_argument(
        "--train-speakers",
        type=Path,
        required=True,
        help="Speaker list for the training split, one speaker ID per line.",
    )
    parser.add_argument(
        "--val-speakers",
        type=Path,
        required=True,
        help="Speaker list for the validation split, one speaker ID per line.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="fisher",
        help="Dataset name stored in each LUAR JSONL row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for LUAR JSONLs, speaker maps, and metadata.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="ascii",
        choices=["ascii", "utf-8"],
        help="Transcript encoding policy. Use ascii to match the paper preprocessing.",
    )
    return parser.parse_args()


def load_speaker_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Speaker list not found: {path}")
    speakers = [line.strip() for line in path.open("r", encoding="utf-8") if line.strip()]
    if not speakers:
        raise ValueError(f"Speaker list is empty: {path}")
    return speakers


def normalize_text(text: str, encoding: str) -> str:
    text = text.strip()
    if encoding == "ascii":
        return text.encode("ascii", "ignore").decode().strip()
    return text


def parse_audio_text_line(raw: str, line_num: int, source: Path) -> tuple[str, str] | None:
    """Return speaker_id and transcript from a supported audio-text manifest row.

    Supported formats:

    - pipe-delimited: utterance_id|audio_path|transcript|speaker_id
    - whitespace-delimited: speaker_id audio_path transcript text
    """

    line = raw.rstrip("\n")
    if not line.strip():
        return None

    pipe_parts = line.split("|")
    if len(pipe_parts) == 4:
        _, _, transcript, speaker_id = pipe_parts
        return speaker_id.strip(), transcript.strip()

    space_parts = line.split(maxsplit=2)
    if len(space_parts) >= 3:
        speaker_id, _, transcript = space_parts
        return speaker_id.strip(), transcript.strip()

    raise ValueError(
        f"Bad line {line_num} in {source}: expected 'utt|path|text|speaker' "
        "or 'speaker path transcript'."
    )


def collect_utterances(source: Path, keep_speakers: set[str], encoding: str) -> dict[str, list[str]]:
    """Collect transcripts for the requested speaker IDs."""

    if not source.is_file():
        raise FileNotFoundError(f"Audio-text source manifest not found: {source}")

    utterances: dict[str, list[str]] = {speaker_id: [] for speaker_id in keep_speakers}
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_num, raw in enumerate(handle, start=1):
            parsed = parse_audio_text_line(raw, line_num, source)
            if parsed is None:
                continue
            speaker_id, transcript = parsed
            if speaker_id not in utterances:
                continue
            text = normalize_text(transcript, encoding)
            if text:
                utterances[speaker_id].append(text)
    return utterances


def write_jsonl(path: Path, dataset_name: str, ordered_speakers: list[str], utterances: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for author_id, speaker_id in enumerate(ordered_speakers):
            syms = utterances.get(speaker_id, [])
            if not syms:
                raise ValueError(f"Speaker {speaker_id} has no transcripts for {path}.")
            row = {
                "dataset": dataset_name,
                "author_id": author_id,
                "syms": syms,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_validation_query_target_split(val_jsonl: Path, output_dir: Path) -> tuple[Path, Path, int]:
    queries_path = output_dir / "validation_queries.jsonl"
    targets_path = output_dir / "validation_targets.jsonl"
    rows_written = 0

    with val_jsonl.open("r", encoding="utf-8") as source, queries_path.open("w", encoding="utf-8") as queries, targets_path.open(
        "w", encoding="utf-8"
    ) as targets:
        for raw in source:
            row = json.loads(raw)
            syms = row["syms"]
            if len(syms) < 2:
                continue
            split_idx = len(syms) // 2
            query = {
                "author_id": row["author_id"],
                "syms": syms[:split_idx],
                "dataset": row["dataset"],
                "action_type": [],
            }
            target = {
                "author_id": row["author_id"],
                "syms": syms[split_idx:],
                "dataset": row["dataset"],
                "action_type": [],
            }
            queries.write(json.dumps(query, ensure_ascii=False) + "\n")
            targets.write(json.dumps(target, ensure_ascii=False) + "\n")
            rows_written += 1

    return queries_path, targets_path, rows_written


def write_speaker_map(path: Path, ordered_speakers: list[str]) -> None:
    payload = [{"author_id": index, "speaker_id": speaker_id} for index, speaker_id in enumerate(ordered_speakers)]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def utterance_count(utterances: dict[str, list[str]]) -> int:
    return sum(len(values) for values in utterances.values())


def main() -> None:
    args = parse_args()

    train_source = args.train_source.expanduser().resolve()
    val_source = args.val_source.expanduser().resolve()
    train_speakers_path = args.train_speakers.expanduser().resolve()
    val_speakers_path = args.val_speakers.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_speakers = load_speaker_list(train_speakers_path)
    val_speakers = load_speaker_list(val_speakers_path)

    overlap = set(train_speakers) & set(val_speakers)
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Train and validation speaker lists overlap for {len(overlap)} speakers. Examples: {examples}")

    train_utterances = collect_utterances(train_source, set(train_speakers), args.encoding)
    val_utterances = collect_utterances(val_source, set(val_speakers), args.encoding)

    train_jsonl = output_dir / "train.jsonl"
    val_jsonl = output_dir / "val.jsonl"
    train_map = output_dir / "train_speaker_map.json"
    val_map = output_dir / "val_speaker_map.json"
    metadata_path = output_dir / "metadata.json"

    write_jsonl(train_jsonl, args.dataset_name, train_speakers, train_utterances)
    write_jsonl(val_jsonl, args.dataset_name, val_speakers, val_utterances)
    validation_queries, validation_targets, validation_query_target_rows = write_validation_query_target_split(
        val_jsonl, output_dir
    )
    write_speaker_map(train_map, train_speakers)
    write_speaker_map(val_map, val_speakers)

    metadata = {
        "dataset_name": args.dataset_name,
        "encoding": args.encoding,
        "train_source": str(train_source),
        "validation_source": str(val_source),
        "train_speakers_file": str(train_speakers_path),
        "validation_speakers_file": str(val_speakers_path),
        "counts": {
            "train_speakers": len(train_speakers),
            "validation_speakers": len(val_speakers),
            "train_transcripts": utterance_count(train_utterances),
            "validation_transcripts": utterance_count(val_utterances),
            "validation_query_target_rows": validation_query_target_rows,
        },
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "validation_queries": str(validation_queries),
            "validation_targets": str(validation_targets),
            "train_speaker_map": str(train_map),
            "validation_speaker_map": str(val_map),
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
