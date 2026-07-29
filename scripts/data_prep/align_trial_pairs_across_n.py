#!/usr/bin/env python3
"""Create aligned trial JSONLs across multiple N values from a larger anchor trial set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from random import Random


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for aligned multi-N trial generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned N-trial JSONLs that preserve the same enroll/test speaker pairs "
            "across different numbers of utterances per side."
        )
    )
    parser.add_argument("--anchor-jsonl", type=Path, required=True, help="Source trial JSONL, typically the highest-N file.")
    parser.add_argument("--n-values", type=int, nargs="+", required=True, help="Output N values, for example: 5 10 15")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to the anchor file directory.")
    parser.add_argument(
        "--output-pattern",
        type=str,
        default=None,
        help="Output pattern with {N}. Defaults to '<anchor_stem>_aligned_N{N}.jsonl'.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "random_nested"],
        default="random_nested",
        help="Subset policy for choosing utterances within each side.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic subset selection.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed rows instead of skipping them.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output files.")
    return parser.parse_args()


def stable_row_seed(base_seed: int, row_index: int, side: str) -> int:
    """Create a deterministic side-specific seed for a given row."""
    digest = hashlib.md5(f"{base_seed}:{row_index}:{side}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def ordered_subset_indices(total: int, n_value: int, *, selection: str, seed: int) -> list[int]:
    """Choose utterance indices while preserving original temporal order."""
    if n_value > total:
        raise ValueError(f"Requested n={n_value} exceeds available items={total}")
    if selection == "first":
        return list(range(n_value))
    rng = Random(seed)
    perm = list(range(total))
    rng.shuffle(perm)
    return sorted(perm[:n_value])


def slice_side(side_obj: dict, idxs: list[int]) -> dict:
    """Subselect a trial side while keeping paths, transcripts, and timestamps aligned."""
    out = copy.deepcopy(side_obj)
    for key in ("paths", "transcripts", "word_timestamps"):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = [value[i] for i in idxs]
    return out


def validate_row_shape(row: dict, max_n: int) -> tuple[int, int]:
    """Validate that a trial row has enough utterances on both sides."""
    enroll = row.get("enroll") or {}
    test = row.get("test") or {}
    e_paths = enroll.get("paths")
    t_paths = test.get("paths")
    if not isinstance(e_paths, list) or not isinstance(t_paths, list):
        raise ValueError("Missing enroll.paths/test.paths list.")
    if len(e_paths) < max_n or len(t_paths) < max_n:
        raise ValueError(f"Not enough utterances for max_n={max_n}: enroll={len(e_paths)} test={len(t_paths)}.")
    return len(e_paths), len(t_paths)


def default_pattern(anchor: Path) -> str:
    """Construct the default aligned output filename pattern."""
    return f"{anchor.stem}_aligned_N{{N}}{anchor.suffix}"


def main() -> None:
    """Create aligned lower-N trial files from an anchor higher-N trial file."""
    args = parse_args()
    if not args.anchor_jsonl.exists():
        raise FileNotFoundError(args.anchor_jsonl)

    n_values = sorted(set(int(value) for value in args.n_values))
    max_n = max(n_values)
    out_dir = args.output_dir or args.anchor_jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = args.output_pattern or default_pattern(args.anchor_jsonl)

    out_paths: dict[int, Path] = {}
    for n_value in n_values:
        out_path = out_dir / pattern.replace("{N}", str(n_value))
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(out_path)
        out_paths[n_value] = out_path

    writers = {n_value: out_paths[n_value].open("w", encoding="utf-8") for n_value in n_values}
    kept = 0
    skipped = 0
    try:
        with args.anchor_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
            for row_index, raw in enumerate(handle):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    e_len, t_len = validate_row_shape(row, max_n=max_n)
                except Exception:
                    if args.strict:
                        raise
                    skipped += 1
                    continue

                e_seed = stable_row_seed(args.seed, row_index, "enroll")
                t_seed = stable_row_seed(args.seed, row_index, "test")
                for n_value in n_values:
                    out_row = dict(row)
                    out_row["enroll"] = slice_side(
                        row["enroll"],
                        ordered_subset_indices(e_len, n_value, selection=args.selection, seed=e_seed),
                    )
                    out_row["test"] = slice_side(
                        row["test"],
                        ordered_subset_indices(t_len, n_value, selection=args.selection, seed=t_seed),
                    )
                    writers[n_value].write(json.dumps(out_row, ensure_ascii=False) + "\n")
                kept += 1
    finally:
        for writer in writers.values():
            writer.close()

    print("Done.")
    print(f"Anchor: {args.anchor_jsonl}")
    print(f"Rows kept: {kept}")
    print(f"Rows skipped: {skipped}")
    for n_value in n_values:
        print(f"N={n_value} -> {out_paths[n_value]}")


if __name__ == "__main__":
    main()
