#!/usr/bin/env python3
"""Attach transcripts and word timestamps to audio-only validation trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build audio+text validation assets from audio-only trials and Whisper-style "
            "transcription JSONL shards."
        )
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=None,
        help="Directory containing source trial JSONL files.",
    )
    parser.add_argument(
        "--trial-jsonl",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit source trial JSONL file(s). If set, --trial-dir and --trial-glob are ignored.",
    )
    parser.add_argument(
        "--trial-glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern used with --trial-dir.",
    )
    parser.add_argument(
        "--transcription-dir",
        type=Path,
        default=None,
        help="Directory containing Whisper transcription JSONL shards.",
    )
    parser.add_argument(
        "--transcription-jsonl",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit Whisper transcription JSONL shard(s). If set, --transcription-dir is ignored.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root for generated trials, manifests, and reports.",
    )
    parser.add_argument(
        "--skip-empty-transcript-manifest",
        action="store_true",
        help="Skip manifest entries with empty/missing transcripts.",
    )
    return parser.parse_args()


def side_paths(side: dict[str, Any]) -> list[str]:
    """Return audio paths from supported trial side schemas."""

    for key in ("paths", "audio_paths"):
        value = side.get(key)
        if isinstance(value, list):
            return [str(path) for path in value]
    if "path" in side:
        return [str(side["path"])]
    return []


def flatten_words(segments: Any) -> list[dict[str, Any]]:
    """Flatten Whisper segments into per-utterance word timestamps."""

    words_out: list[dict[str, Any]] = []
    if not isinstance(segments, list):
        return words_out
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            words_out.append(
                {
                    "word": str(word.get("word", "")),
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                }
            )
    return words_out


def sanitize_manifest_text(text: str) -> str:
    return str(text).replace("|", " ").strip()


def select_trial_files(args: argparse.Namespace) -> list[Path]:
    if args.trial_jsonl:
        trial_files = [path.expanduser().resolve() for path in args.trial_jsonl]
    else:
        if args.trial_dir is None:
            raise ValueError("Provide either --trial-jsonl or --trial-dir.")
        trial_dir = args.trial_dir.expanduser().resolve()
        trial_files = sorted(trial_dir.glob(args.trial_glob))

    missing = [str(path) for path in trial_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trial JSONL file(s): {missing}")
    if not trial_files:
        raise FileNotFoundError("No trial JSONL files were selected.")
    return trial_files


def select_transcription_files(args: argparse.Namespace) -> list[Path]:
    if args.transcription_jsonl:
        shard_files = [path.expanduser().resolve() for path in args.transcription_jsonl]
    else:
        if args.transcription_dir is None:
            raise ValueError("Provide either --transcription-jsonl or --transcription-dir.")
        transcription_dir = args.transcription_dir.expanduser().resolve()
        shard_files = sorted(transcription_dir.glob("*.jsonl"))

    missing = [str(path) for path in shard_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing transcription JSONL file(s): {missing}")
    if not shard_files:
        raise FileNotFoundError("No transcription JSONL files were selected.")
    return shard_files


def load_trials_and_needed_basenames(trial_files: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    trials_by_name: dict[str, list[dict[str, Any]]] = {}
    needed: set[str] = set()

    for trial_file in trial_files:
        rows: list[dict[str, Any]] = []
        with trial_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)
                for side_name in ("enroll", "test"):
                    for audio_path in side_paths(row.get(side_name, {})):
                        needed.add(Path(audio_path).name)
        trials_by_name[trial_file.stem] = rows

    return trials_by_name, needed


def build_transcript_lookup(shard_files: list[Path], needed_basenames: set[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for shard_file in shard_files:
        with shard_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                filename = row.get("file")
                if not isinstance(filename, str):
                    filename = row.get("audio")
                if not isinstance(filename, str):
                    filename = row.get("path")
                if not isinstance(filename, str):
                    continue

                basename = Path(filename).name
                if basename not in needed_basenames or basename in lookup:
                    continue

                lookup[basename] = {
                    "text": str(row.get("text", "")),
                    "words": flatten_words(row.get("segments")),
                }
                if len(lookup) == len(needed_basenames):
                    return lookup
    return lookup


def paths_to_text_and_words(
    paths: list[str],
    transcript_lookup: dict[str, dict[str, Any]],
) -> tuple[list[str], list[list[dict[str, Any]]], int]:
    transcripts: list[str] = []
    words_by_utterance: list[list[dict[str, Any]]] = []
    missing = 0

    for audio_path in paths:
        record = transcript_lookup.get(Path(audio_path).name)
        if record is None:
            transcripts.append("")
            words_by_utterance.append([])
            missing += 1
            continue
        transcripts.append(str(record.get("text", "")))
        words_by_utterance.append(record.get("words", []))

    return transcripts, words_by_utterance, missing


def collect_unique_paths(trials: list[dict[str, Any]]) -> dict[str, str]:
    path_to_speaker: dict[str, str] = {}
    for row in trials:
        for side_name in ("enroll", "test"):
            side = row.get(side_name, {})
            speaker = str(side.get("speaker", ""))
            for audio_path in side_paths(side):
                path_to_speaker.setdefault(audio_path, speaker)
    return path_to_speaker


def write_manifest_pair(
    path_to_speaker: dict[str, str],
    transcript_lookup: dict[str, dict[str, Any]],
    soft_path: Path,
    hard_path: Path,
    skip_empty: bool,
) -> dict[str, int]:
    soft_path.parent.mkdir(parents=True, exist_ok=True)
    hard_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows_written": 0,
        "missing_in_lookup": 0,
        "empty_text_rows": 0,
        "skipped_empty_rows": 0,
    }

    with soft_path.open("w", encoding="utf-8") as soft_file, hard_path.open("w", encoding="utf-8") as hard_file:
        for audio_path in sorted(path_to_speaker):
            speaker = path_to_speaker[audio_path]
            basename = Path(audio_path).name
            utterance_id = Path(basename).stem
            record = transcript_lookup.get(basename)
            if record is None:
                stats["missing_in_lookup"] += 1
                text = ""
                words: list[dict[str, Any]] = []
            else:
                text = sanitize_manifest_text(str(record.get("text", "")))
                words = record.get("words", [])

            if not text:
                stats["empty_text_rows"] += 1
                if skip_empty:
                    stats["skipped_empty_rows"] += 1
                    continue

            soft_file.write(f"{utterance_id}|{audio_path}|{text}|{speaker}\n")
            hard_file.write(
                f"{speaker}|{audio_path}|{text}|"
                + json.dumps(words, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stats["rows_written"] += 1

    return stats


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    trials_out_dir = out_dir / "trials"
    manifests_soft_dir = out_dir / "manifests" / "soft"
    manifests_hard_dir = out_dir / "manifests" / "hard_inline"
    reports_dir = out_dir / "reports"
    trials_out_dir.mkdir(parents=True, exist_ok=True)
    manifests_soft_dir.mkdir(parents=True, exist_ok=True)
    manifests_hard_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    trial_files = select_trial_files(args)
    transcription_files = select_transcription_files(args)
    trials_by_name, needed_basenames = load_trials_and_needed_basenames(trial_files)
    transcript_lookup = build_transcript_lookup(transcription_files, needed_basenames)

    matched_subset_jsonl = reports_dir / "matched_transcripts_subset.jsonl"
    with matched_subset_jsonl.open("w", encoding="utf-8") as handle:
        for filename in sorted(transcript_lookup):
            record = transcript_lookup[filename]
            handle.write(
                json.dumps(
                    {"file": filename, "text": record["text"], "words": record["words"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    missing_names = sorted(needed_basenames - set(transcript_lookup))
    missing_list = reports_dir / "missing_transcript_filenames.txt"
    missing_list.write_text("".join(f"{name}\n" for name in missing_names), encoding="utf-8")

    summary: dict[str, Any] = {
        "trial_files": [str(path) for path in trial_files],
        "transcription_jsonl_count": len(transcription_files),
        "needed_unique_filenames": len(needed_basenames),
        "matched_transcripts": len(transcript_lookup),
        "missing_transcripts": len(missing_names),
        "skip_empty_transcript_manifest": bool(args.skip_empty_transcript_manifest),
        "trials": {},
        "manifests": {"per_trial": {}},
        "reports": {
            "matched_transcripts_subset": str(matched_subset_jsonl),
            "missing_transcript_filenames": str(missing_list),
        },
    }

    all_soft_trials: list[dict[str, Any]] = []
    for stem, rows in trials_by_name.items():
        soft_trial_path = trials_out_dir / f"{stem}_audio_text.jsonl"
        hard_trial_path = trials_out_dir / f"{stem}_audio_text_hard.jsonl"

        trial_count = 0
        missing_slots = 0
        with soft_trial_path.open("w", encoding="utf-8") as soft_file, hard_trial_path.open(
            "w", encoding="utf-8"
        ) as hard_file:
            for row in rows:
                trial_count += 1
                enroll = row.get("enroll", {})
                test = row.get("test", {})
                enroll_paths = side_paths(enroll)
                test_paths = side_paths(test)

                enroll_text, enroll_words, enroll_missing = paths_to_text_and_words(enroll_paths, transcript_lookup)
                test_text, test_words, test_missing = paths_to_text_and_words(test_paths, transcript_lookup)
                missing_slots += enroll_missing + test_missing

                soft_row = {
                    "label": int(row.get("label", 0)),
                    "enroll": {
                        "speaker": str(enroll.get("speaker", "")),
                        "audio_paths": enroll_paths,
                        "transcripts": enroll_text,
                    },
                    "test": {
                        "speaker": str(test.get("speaker", "")),
                        "audio_paths": test_paths,
                        "transcripts": test_text,
                    },
                }
                hard_row = {
                    "label": int(row.get("label", 0)),
                    "enroll": {
                        "speaker": str(enroll.get("speaker", "")),
                        "audio_paths": enroll_paths,
                        "transcripts": enroll_text,
                        "word_timestamps": enroll_words,
                    },
                    "test": {
                        "speaker": str(test.get("speaker", "")),
                        "audio_paths": test_paths,
                        "transcripts": test_text,
                        "word_timestamps": test_words,
                    },
                }
                soft_file.write(json.dumps(soft_row, ensure_ascii=False) + "\n")
                hard_file.write(json.dumps(hard_row, ensure_ascii=False) + "\n")
                all_soft_trials.append(soft_row)

        summary["trials"][stem] = {
            "source": str(next(path for path in trial_files if path.stem == stem)),
            "soft": str(soft_trial_path),
            "hard": str(hard_trial_path),
            "trials": trial_count,
            "missing_transcript_slots": missing_slots,
        }

        path_to_speaker = collect_unique_paths(rows)
        soft_manifest = manifests_soft_dir / f"{stem}_audio_text_soft_manifest.txt"
        hard_manifest = manifests_hard_dir / f"{stem}_audio_text_hard_inline_manifest.txt"
        summary["manifests"]["per_trial"][stem] = {
            "soft": str(soft_manifest),
            "hard_inline": str(hard_manifest),
            **write_manifest_pair(
                path_to_speaker,
                transcript_lookup,
                soft_manifest,
                hard_manifest,
                skip_empty=args.skip_empty_transcript_manifest,
            ),
        }

    all_path_to_speaker = collect_unique_paths(all_soft_trials)
    all_soft_manifest = manifests_soft_dir / "all_audio_text_soft_manifest.txt"
    all_hard_manifest = manifests_hard_dir / "all_audio_text_hard_inline_manifest.txt"
    summary["manifests"]["all"] = {
        "soft": str(all_soft_manifest),
        "hard_inline": str(all_hard_manifest),
        **write_manifest_pair(
            all_path_to_speaker,
            transcript_lookup,
            all_soft_manifest,
            all_hard_manifest,
            skip_empty=args.skip_empty_transcript_manifest,
        ),
    }

    summary_path = reports_dir / "build_audio_text_val_assets_summary.json"
    summary["reports"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
