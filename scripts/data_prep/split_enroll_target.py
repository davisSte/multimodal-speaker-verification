#!/usr/bin/env python3
"""Split a speaker/audio subset into per-speaker enroll and target files."""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for enroll/target splitting."""
    parser = argparse.ArgumentParser(
        description="Split each speaker's utterances into enroll and target subsets."
    )
    parser.add_argument("--audio-subset", type=Path, required=True, help="Input file with lines: '<speaker_id> <audio_path>'.")
    parser.add_argument("--enroll-ratio", type=float, default=0.3, help="Fraction of utterances assigned to enroll.")
    parser.add_argument("--enroll-file", type=Path, required=True, help="Output enroll subset file.")
    parser.add_argument("--target-file", type=Path, required=True, help="Output target subset file.")
    parser.add_argument("--stats-file", type=Path, required=True, help="Output per-speaker split statistics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-speaker shuffling.")
    return parser.parse_args()


def load_subset(path: Path) -> dict[str, list[str]]:
    """Load a speaker/audio text file into a speaker-to-utterances mapping."""
    spk2utts: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            speaker, wav_path = parts
            spk2utts[speaker].append(wav_path)
    return spk2utts


def main() -> None:
    """Split each speaker's utterances and write enroll/target subset files."""
    args = parse_args()
    if not args.audio_subset.exists():
        raise FileNotFoundError(args.audio_subset)
    if not 0.0 < args.enroll_ratio < 1.0:
        raise ValueError("--enroll-ratio must be between 0 and 1.")

    args.enroll_file.parent.mkdir(parents=True, exist_ok=True)
    args.target_file.parent.mkdir(parents=True, exist_ok=True)
    args.stats_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    spk2utts = load_subset(args.audio_subset)

    with (
        args.enroll_file.open("w", encoding="utf-8") as enroll_handle,
        args.target_file.open("w", encoding="utf-8") as target_handle,
        args.stats_file.open("w", encoding="utf-8") as stats_handle,
    ):
        for speaker, utterances in sorted(spk2utts.items()):
            total = len(utterances)
            if total < 2:
                continue
            shuffled = list(utterances)
            rng.shuffle(shuffled)
            num_enroll = max(1, math.floor(total * args.enroll_ratio))
            enroll_utts = shuffled[:num_enroll]
            target_utts = shuffled[num_enroll:]
            if not target_utts:
                continue
            for wav_path in enroll_utts:
                enroll_handle.write(f"{speaker} {wav_path}\n")
            for wav_path in target_utts:
                target_handle.write(f"{speaker} {wav_path}\n")
            stats_handle.write(
                f"{speaker} total={total} enroll={len(enroll_utts)} target={len(target_utts)}\n"
            )

    print(f"Enroll: {args.enroll_file}")
    print(f"Target: {args.target_file}")
    print(f"Stats: {args.stats_file}")


if __name__ == "__main__":
    main()
