#!/usr/bin/env python3
"""Build an audio-only speaker/path manifest from one or more trial JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for audio-only manifest generation."""
    parser = argparse.ArgumentParser(description="Build an audio-only manifest from trial JSONL files.")
    parser.add_argument("--trial-dir", type=Path, required=True, help="Directory containing trial JSONLs.")
    parser.add_argument("--trial-glob", type=str, default="*.jsonl", help="Glob pattern used inside --trial-dir.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output speaker/path manifest.")
    return parser.parse_args()


def iter_paths(block: dict) -> list[str]:
    """Return the list of paths stored in a trial-side dictionary."""
    if isinstance(block.get("paths"), list):
        return [str(path) for path in block["paths"]]
    if block.get("path") is not None:
        return [str(block["path"])]
    return []


def main() -> None:
    """Extract unique speaker/path pairs from trial files and write a manifest."""
    args = parse_args()
    trial_files = sorted(args.trial_dir.glob(args.trial_glob))
    if not trial_files:
        raise FileNotFoundError(f"No trial files matched: dir={args.trial_dir} glob={args.trial_glob}")

    path_to_speaker: dict[str, str] = {}
    total_rows = 0
    for trial_file in trial_files:
        with trial_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                total_rows += 1
                for side in ("enroll", "test"):
                    block = row.get(side, {})
                    speaker = str(block.get("speaker", ""))
                    for path in iter_paths(block):
                        prev = path_to_speaker.get(path)
                        if prev is not None and prev != speaker:
                            raise ValueError(f"Speaker mismatch for {path}: {prev} vs {speaker}")
                        path_to_speaker[path] = speaker

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as handle:
        for speaker, path in sorted((speaker, path) for path, speaker in path_to_speaker.items()):
            handle.write(f"{speaker} {path}\n")

    print(f"Trial files: {len(trial_files)}")
    print(f"Trial rows: {total_rows}")
    print(f"Manifest rows: {len(path_to_speaker)}")
    print(f"Manifest: {args.manifest_out}")
