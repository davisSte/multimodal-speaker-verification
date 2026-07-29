#!/usr/bin/env python3
"""Sample a fixed number of speakers from a master validation speaker/audio list."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for speaker subset creation."""
    parser = argparse.ArgumentParser(
        description="Sample K speakers from a master speaker/audio list and keep all utterances for each selected speaker."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Input file with lines: '<speaker_id> <audio_path>'.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output file containing the sampled speaker subset.")
    parser.add_argument("--num-speakers", type=int, required=True, help="Number of speakers to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for speaker sampling.")
    return parser.parse_args()


def main() -> None:
    """Sample speakers and write all of their utterances to the output file."""
    args = parse_args()
    if args.num_speakers <= 0:
        raise ValueError("--num-speakers must be > 0")
    if not args.input_file.exists():
        raise FileNotFoundError(args.input_file)

    spk2lines = defaultdict(list)
    with args.input_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            speaker = line.split(maxsplit=1)[0]
            spk2lines[speaker].append(line)

    speakers = sorted(spk2lines)
    if args.num_speakers > len(speakers):
        raise ValueError(f"Requested {args.num_speakers} speakers but only {len(speakers)} are available.")

    rng = random.Random(args.seed)
    selected = set(rng.sample(speakers, args.num_speakers))

    output_lines: list[str] = []
    for speaker in speakers:
        if speaker in selected:
            output_lines.extend(spk2lines[speaker])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for line in output_lines:
            handle.write(line + "\n")

    print(f"Wrote: {args.output_file}")
    print(f"Selected speakers: {len(selected)}")
    print(f"Rows written: {len(output_lines)}")


if __name__ == "__main__":
    main()
