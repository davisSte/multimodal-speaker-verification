#!/usr/bin/env python3
"""Build a Fisher utterance manifest from pre-segmented wav files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a speaker_id audio_path manifest from Fisher utterance wavs. "
            "Speaker IDs are read from filenames such as fe_03_00001_A_103_2602.wav."
        )
    )
    parser.add_argument("--utterance-root", type=Path, required=True, help="Root containing split Fisher wavs.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output manifest path.")
    parser.add_argument("--metadata-out", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--relative-to", type=Path, help="Write paths relative to this directory.")
    parser.add_argument("--extension", default=".wav", help="Audio extension to include.")
    return parser.parse_args()


def speaker_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Cannot infer Fisher speaker ID from filename: {path}")
    speaker_id = stem.rsplit("_", maxsplit=1)[-1]
    if not speaker_id:
        raise ValueError(f"Empty speaker ID inferred from filename: {path}")
    return speaker_id


def output_path(path: Path, relative_to: Path | None) -> str:
    resolved = path.resolve()
    if relative_to is None:
        return str(resolved)
    return str(resolved.relative_to(relative_to.resolve()))


def main() -> None:
    args = parse_args()
    utterance_root = args.utterance_root.expanduser().resolve()
    extension = args.extension if args.extension.startswith(".") else f".{args.extension}"
    wavs = sorted(path for path in utterance_root.rglob(f"*{extension}") if path.is_file())

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    speakers: set[str] = set()
    with args.manifest_out.open("w", encoding="utf-8") as handle:
        for wav in wavs:
            speaker_id = speaker_from_filename(wav)
            speakers.add(speaker_id)
            handle.write(f"{speaker_id} {output_path(wav, args.relative_to)}\n")

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(
            json.dumps(
                {
                    "utterance_root": str(utterance_root),
                    "manifest_out": str(args.manifest_out.expanduser().resolve()),
                    "num_utterances": len(wavs),
                    "num_speakers": len(speakers),
                    "extension": extension,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {len(wavs)} utterances from {len(speakers)} speakers to {args.manifest_out}")


if __name__ == "__main__":
    main()
