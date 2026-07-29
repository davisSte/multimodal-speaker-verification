#!/usr/bin/env python3
"""Create multi-utterance positive and negative trial JSONL files from speaker subsets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for trial generation."""
    parser = argparse.ArgumentParser(
        description="Create speaker-verification trial JSONL files from enroll and target subsets."
    )
    parser.add_argument("--enroll-file", type=Path, required=True, help="Enroll subset file.")
    parser.add_argument("--target-file", type=Path, required=True, help="Target subset file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output trial JSONL.")
    parser.add_argument("--num-utts", type=int, required=True, help="Number of utterances per enroll/test side.")
    parser.add_argument("--trials-per-speaker", type=int, default=500, help="Positive and negative trials per speaker.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trial sampling.")
    parser.add_argument(
        "--path-prefix-from",
        type=str,
        default="",
        help="Optional input path prefix to rewrite.",
    )
    parser.add_argument(
        "--path-prefix-to",
        type=str,
        default="",
        help="Optional replacement prefix when rewriting paths.",
    )
    parser.add_argument(
        "--write-remapped-subsets-dir",
        type=Path,
        default=None,
        help="Optional directory to write remapped enroll/target subset files.",
    )
    return parser.parse_args()


def remap_path(path: str, src_prefix: str, dst_prefix: str) -> str:
    """Rewrite a legacy path prefix when requested."""
    if src_prefix and dst_prefix and path.startswith(src_prefix):
        suffix = path[len(src_prefix):].lstrip("/")
        return f"{dst_prefix.rstrip('/')}/{suffix}"
    return path


def load_subset(file_path: Path, src_prefix: str, dst_prefix: str) -> dict[str, list[str]]:
    """Load a speaker/audio subset file into a speaker-to-utterances mapping."""
    spk2utts: dict[str, list[str]] = defaultdict(list)
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            speaker, path = line.split(maxsplit=1)
            spk2utts[speaker].append(remap_path(path, src_prefix=src_prefix, dst_prefix=dst_prefix))
    return spk2utts


def dump_subset(file_path: Path, spk2utts: dict[str, list[str]]) -> None:
    """Write a speaker-to-utterances mapping back to a text subset file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for speaker, paths in spk2utts.items():
            for wav_path in paths:
                handle.write(f"{speaker} {wav_path}\n")


def generate_trials(
    spk2enroll: dict[str, list[str]],
    spk2target: dict[str, list[str]],
    *,
    n_utts: int,
    trials_per_speaker: int,
    seed: int,
) -> list[dict]:
    """Create positive and negative trial rows for each eligible speaker."""
    rng = random.Random(seed)
    trials: list[dict] = []
    speakers = list(spk2enroll.keys())

    for speaker in tqdm(speakers, desc="Generating trials"):
        enroll_utts = spk2enroll[speaker]
        target_utts = spk2target.get(speaker, [])
        if len(enroll_utts) < n_utts or len(target_utts) < n_utts:
            continue

        for _ in range(trials_per_speaker):
            trials.append(
                {
                    "label": 1,
                    "enroll": {"speaker": speaker, "paths": rng.sample(enroll_utts, n_utts)},
                    "test": {"speaker": speaker, "paths": rng.sample(target_utts, n_utts)},
                }
            )

        eligible_neg = [spk for spk in speakers if spk != speaker and len(spk2target.get(spk, [])) >= n_utts]
        for _ in range(trials_per_speaker):
            if not eligible_neg:
                break
            neg_speaker = rng.choice(eligible_neg)
            trials.append(
                {
                    "label": 0,
                    "enroll": {"speaker": speaker, "paths": rng.sample(enroll_utts, n_utts)},
                    "test": {"speaker": neg_speaker, "paths": rng.sample(spk2target[neg_speaker], n_utts)},
                }
            )

    return trials


def main() -> None:
    """Load enroll/target subsets, generate trials, and write JSONL output."""
    args = parse_args()
    if not args.enroll_file.exists():
        raise FileNotFoundError(args.enroll_file)
    if not args.target_file.exists():
        raise FileNotFoundError(args.target_file)

    spk2enroll = load_subset(args.enroll_file, args.path_prefix_from, args.path_prefix_to)
    spk2target = load_subset(args.target_file, args.path_prefix_from, args.path_prefix_to)

    if args.write_remapped_subsets_dir is not None:
        args.write_remapped_subsets_dir.mkdir(parents=True, exist_ok=True)
        dump_subset(args.write_remapped_subsets_dir / "enroll_subsets.txt", spk2enroll)
        dump_subset(args.write_remapped_subsets_dir / "target_subsets.txt", spk2target)

    trials = generate_trials(
        spk2enroll,
        spk2target,
        n_utts=args.num_utts,
        trials_per_speaker=args.trials_per_speaker,
        seed=args.seed,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for row in trials:
            handle.write(json.dumps(row) + "\n")

    print(f"Saved {len(trials)} trials to {args.output_file}")


if __name__ == "__main__":
    main()
