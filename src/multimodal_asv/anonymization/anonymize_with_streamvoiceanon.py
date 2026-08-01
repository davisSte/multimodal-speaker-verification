#!/usr/bin/env python3
"""Run StreamVoiceAnon anonymization from an explicit source/reference pair file.

The pair file is the reproducibility anchor. Each non-empty line must contain:

    source_wav|source_speaker|reference_wav|reference_speaker

For each assigned line, this script writes:

    <out_root>/<source_speaker>/<source_stem>__to__<reference_speaker>__<reference_stem>.wav
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multimodal_asv.anonymization.streamvoiceanon import (
    StreamVoiceAnonBatchRunner,
    write_metadata,
    write_original_to_anonymized_mapping,
)

DEFAULT_STREAMVOICEANON_ROOT = (
    Path(__file__).resolve().parent
    / "StreamVoiceAnon"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anonymize audio with StreamVoiceAnon from source/reference pairs."
    )
    parser.add_argument(
        "--pairs-file",
        required=True,
        help="Lines: source_wav|source_speaker|reference_wav|reference_speaker",
    )
    parser.add_argument("--out-root", required=True, help="Directory where anonymized wavs are written.")
    parser.add_argument(
        "--streamvoiceanon-root",
        default=None,
        help="Path to the StreamVoiceAnon checkout. Defaults to the bundled source under src/multimodal_asv/anonymization/StreamVoiceAnon.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/config_firefly_arvcasr_8192_delay0_8.yaml",
        help="StreamVoiceAnon config path. Relative paths are resolved under --streamvoiceanon-root.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="pretrained_checkpoints/dual_ar_delay_0_8.pth",
        help="StreamVoiceAnon checkpoint path. Relative paths are resolved under --streamvoiceanon-root.",
    )
    parser.add_argument("--delay", type=int, default=2, help="StreamVoiceAnon inference delay.")
    parser.add_argument("--alpha", type=float, default=1.0, help="StreamVoiceAnon anonymization mixing value.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for this process.")
    parser.add_argument("--compile", action="store_true", help="Enable StreamVoiceAnon AR compilation.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress logging interval.")
    parser.add_argument(
        "--skip-existing-root",
        action="append",
        default=[],
        help="Additional output roots that count as already completed.",
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON file recording the anonymization settings for this run.",
    )
    parser.add_argument(
        "--mapping-out",
        default=None,
        help="Optional JSON file mapping original source wavs to expected anonymized wavs.",
    )
    parser.add_argument(
        "--write-mapping-only",
        action="store_true",
        help="Write --mapping-out from the pair file and exit without loading StreamVoiceAnon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    streamvoiceanon_root = args.streamvoiceanon_root or DEFAULT_STREAMVOICEANON_ROOT

    if args.write_mapping_only:
        if args.mapping_out is None:
            raise ValueError("--write-mapping-only requires --mapping-out.")
        write_original_to_anonymized_mapping(
            mapping_out=args.mapping_out,
            pairs_file=args.pairs_file,
            out_root=args.out_root,
        )
        return

    runner = StreamVoiceAnonBatchRunner(
        streamvoiceanon_root=streamvoiceanon_root,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        compile_ar=args.compile,
    )

    if args.metadata_out is not None:
        write_metadata(
            metadata_out=args.metadata_out,
            pairs_file=args.pairs_file,
            out_root=args.out_root,
            runner=runner,
            delay=args.delay,
            alpha=args.alpha,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            skip_existing_roots=args.skip_existing_root,
        )

    runner.run_pair_file(
        pairs_file=args.pairs_file,
        out_root=args.out_root,
        delay=args.delay,
        alpha=args.alpha,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        skip_existing_roots=args.skip_existing_root,
        log_every=args.log_every,
    )

    if args.mapping_out is not None:
        write_original_to_anonymized_mapping(
            mapping_out=args.mapping_out,
            pairs_file=args.pairs_file,
            out_root=args.out_root,
        )


if __name__ == "__main__":
    main()
