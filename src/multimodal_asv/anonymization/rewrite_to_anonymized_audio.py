"""Rewrite manifests and trial JSONLs to use anonymized audio paths."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite manifests or trial JSONLs by replacing original audio paths with anonymized paths."
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        required=True,
        help="JSON mapping from original audio path to anonymized audio path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input manifest or trial JSONL.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Rewritten output path.")
    parser.add_argument(
        "--format",
        choices=["manifest", "trial-jsonl"],
        required=True,
        help="Input/output file format.",
    )
    parser.add_argument(
        "--missing",
        choices=["error", "keep"],
        default="error",
        help="What to do when an audio path is absent from the mapping.",
    )
    parser.add_argument(
        "--path-resolution",
        choices=["as-is", "absolute", "input-relative"],
        default="as-is",
        help=(
            "How to look up input paths in the mapping. Use absolute for manifests/trials "
            "with absolute paths, input-relative when relative paths are stored next to the input file."
        ),
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Mapping JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mapping JSON must contain an object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def mapping_key(audio_path: str, input_file: Path, path_resolution: str) -> str:
    value = Path(audio_path).expanduser()
    if path_resolution == "as-is":
        return str(value)
    if path_resolution == "absolute":
        return str(value.resolve())
    if path_resolution == "input-relative":
        return str(value if value.is_absolute() else (input_file.parent / value).resolve())
    raise ValueError(f"Unsupported path resolution: {path_resolution}")


def rewrite_audio_path(
    audio_path: str,
    mapping: dict[str, str],
    input_file: Path,
    path_resolution: str,
    missing: str,
) -> tuple[str, bool]:
    key = mapping_key(audio_path, input_file, path_resolution)
    mapped = mapping.get(key)
    if mapped is not None:
        return mapped, True
    if missing == "keep":
        return audio_path, False
    raise KeyError(f"Audio path not found in anonymization mapping: {audio_path} (lookup key: {key})")


def rewrite_manifest(args: argparse.Namespace, mapping: dict[str, str]) -> dict[str, int]:
    stats = {"rows": 0, "rewritten_paths": 0, "kept_missing_paths": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="ignore") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line_num, raw in enumerate(source, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            stats["rows"] += 1

            if "|" in line:
                parts = line.split("|")
                if len(parts) < 2:
                    raise ValueError(f"Bad pipe-delimited manifest row {line_num}: {line!r}")
                new_path, found = rewrite_audio_path(
                    parts[1],
                    mapping,
                    args.input,
                    args.path_resolution,
                    args.missing,
                )
                parts[1] = new_path
                output.write("|".join(parts) + "\n")
            else:
                parts = line.split(maxsplit=2)
                if len(parts) < 2:
                    raise ValueError(f"Bad whitespace manifest row {line_num}: {line!r}")
                new_path, found = rewrite_audio_path(
                    parts[1],
                    mapping,
                    args.input,
                    args.path_resolution,
                    args.missing,
                )
                parts[1] = new_path
                output.write(" ".join(parts) + "\n")

            if found:
                stats["rewritten_paths"] += 1
            else:
                stats["kept_missing_paths"] += 1

    return stats


def rewrite_side(
    side: dict[str, Any],
    mapping: dict[str, str],
    input_file: Path,
    path_resolution: str,
    missing: str,
) -> tuple[dict[str, Any], int, int]:
    rewritten = deepcopy(side)
    rewritten_count = 0
    kept_count = 0

    for key in ("audio_paths", "paths"):
        value = rewritten.get(key)
        if isinstance(value, list):
            new_values = []
            for audio_path in value:
                new_path, found = rewrite_audio_path(str(audio_path), mapping, input_file, path_resolution, missing)
                new_values.append(new_path)
                rewritten_count += int(found)
                kept_count += int(not found)
            rewritten[key] = new_values

    if "path" in rewritten:
        new_path, found = rewrite_audio_path(str(rewritten["path"]), mapping, input_file, path_resolution, missing)
        rewritten["path"] = new_path
        rewritten_count += int(found)
        kept_count += int(not found)

    return rewritten, rewritten_count, kept_count


def rewrite_trial_jsonl(args: argparse.Namespace, mapping: dict[str, str]) -> dict[str, int]:
    stats = {"rows": 0, "rewritten_paths": 0, "kept_missing_paths": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="ignore") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for raw in source:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["rows"] += 1
            rewritten_row = deepcopy(row)

            for side_name in ("enroll", "test"):
                side = rewritten_row.get(side_name)
                if not isinstance(side, dict):
                    continue
                rewritten_side, rewritten_count, kept_count = rewrite_side(
                    side,
                    mapping,
                    args.input,
                    args.path_resolution,
                    args.missing,
                )
                rewritten_row[side_name] = rewritten_side
                stats["rewritten_paths"] += rewritten_count
                stats["kept_missing_paths"] += kept_count

            output.write(json.dumps(rewritten_row, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    mapping = load_mapping(args.mapping_json.expanduser().resolve())

    if args.format == "manifest":
        stats = rewrite_manifest(args, mapping)
    else:
        stats = rewrite_trial_jsonl(args, mapping)

    print(json.dumps({"input": str(args.input), "output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
