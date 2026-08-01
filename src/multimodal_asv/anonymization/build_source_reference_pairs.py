"""Build StreamVoiceAnon source-reference pair files.

The output file is consumed by:

    python -m multimodal_asv.anonymization.anonymize_with_streamvoiceanon

Each output line has:

    source_wav|source_speaker|reference_wav|reference_speaker
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torchaudio


@dataclass
class PairBuildSummary:
    """Counts and settings from source-reference pair creation."""

    source_manifest: str
    reference_root: str
    output_path: str
    seed: int
    min_reference_seconds: float
    source_lines_read: int = 0
    malformed_source_lines: int = 0
    pairs_written: int = 0
    reference_files_found: int = 0
    reference_files_eligible: int = 0
    reference_files_unreadable: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a StreamVoiceAnon pair file from a source manifest and LibriSpeech reference root."
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Source manifest with one utterance per line: '<source_speaker> <source_wav>'.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Reference audio root, for example the LibriSpeech directory containing train-clean-360 and train-other-500.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output pair file: source_wav|source_speaker|reference_wav|reference_speaker.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional JSON file with pair-generation settings and counts.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reference utterance sampling.")
    parser.add_argument(
        "--min-reference-seconds",
        type=float,
        default=4.0,
        help="Minimum reference utterance duration in seconds.",
    )
    return parser.parse_args()


def reference_speaker_id(path: Path, reference_root: Path) -> str:
    """Infer LibriSpeech speaker ID from split/speaker/chapter/file layout."""

    relative = path.resolve().relative_to(reference_root.resolve())
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected LibriSpeech path layout: {path}")
    return relative.parts[1]


def collect_reference_files(reference_root: Path, min_seconds: float, summary: PairBuildSummary) -> list[Path]:
    """Collect reference wav/flac files with duration at least min_seconds."""

    all_paths = sorted(list(reference_root.rglob("*.flac")) + list(reference_root.rglob("*.wav")))
    summary.reference_files_found = len(all_paths)
    if not all_paths:
        raise FileNotFoundError(f"No .flac or .wav files found under reference root: {reference_root}")

    eligible: list[Path] = []
    for path in all_paths:
        try:
            info = torchaudio.info(str(path))
            duration_sec = float(info.num_frames) / float(info.sample_rate)
        except Exception:  # noqa: BLE001
            summary.reference_files_unreadable += 1
            continue
        if duration_sec >= min_seconds:
            eligible.append(path)

    summary.reference_files_eligible = len(eligible)
    if not eligible:
        raise RuntimeError(
            f"No reference files with duration >= {min_seconds:.2f}s under reference root: {reference_root}"
        )
    return eligible


def build_pairs(args: argparse.Namespace) -> PairBuildSummary:
    """Build the source-reference pair file and return a summary."""

    source_manifest = args.source_manifest.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_path = args.out.expanduser().resolve()

    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")

    summary = PairBuildSummary(
        source_manifest=str(source_manifest),
        reference_root=str(reference_root),
        output_path=str(output_path),
        seed=args.seed,
        min_reference_seconds=args.min_reference_seconds,
    )
    rng = random.Random(args.seed)
    reference_paths = collect_reference_files(reference_root, args.min_reference_seconds, summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_manifest.open("r", encoding="utf-8", errors="ignore") as source_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for raw in source_file:
            line = raw.strip()
            if not line:
                continue
            summary.source_lines_read += 1

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                summary.malformed_source_lines += 1
                continue

            source_speaker, source_wav = parts
            reference_wav = reference_paths[rng.randrange(len(reference_paths))]
            reference_speaker = reference_speaker_id(reference_wav, reference_root)
            output_file.write(f"{source_wav}|{source_speaker}|{reference_wav}|{reference_speaker}\n")
            summary.pairs_written += 1

    return summary


def main() -> None:
    args = parse_args()
    summary = build_pairs(args)

    if args.metadata_out is not None:
        metadata_path = args.metadata_out.expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
