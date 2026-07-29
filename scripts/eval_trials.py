#!/usr/bin/env python3
"""Evaluate trained audio or multimodal checkpoints on trial JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from multimodal_asv.config import load_config
from multimodal_asv.evaluation.common import write_tsv
from multimodal_asv.evaluation.trials import (
    evaluate_checkpoint_trials,
    family_side_pooling,
    load_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint-based trial evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate audio and multimodal ASV checkpoints on trial JSONL files. "
            "The model is rebuilt from a paper-facing config and then restored from a training checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Paper-facing experiment config YAML.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Training checkpoint from scripts/train.py.")
    parser.add_argument("--trials-jsonl", type=Path, required=True, help="Trial JSONL with enroll/test sides.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--score-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--tsv-out", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    """Restore a training checkpoint, score trials, and write summary artifacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint_model(config, args.checkpoint_path, device)
    results = evaluate_checkpoint_trials(
        model,
        config,
        trials_jsonl=args.trials_jsonl,
        device=device,
        max_trials=args.max_trials,
        log_every=args.log_every,
    )

    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": results["scores"],
        "labels": results["labels"],
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config),
            "checkpoint_path": str(args.checkpoint_path),
            "trials_jsonl": str(args.trials_jsonl),
            "device": str(device),
            "model_family": config.model.family,
            "side_pooling": family_side_pooling(config),
            "num_trials_total": results["num_trials_total"],
            "num_trials_scored": results["num_trials_scored"],
            "num_trials_skipped_bad": results["num_trials_skipped_bad"],
            "unique_side_count": results["unique_side_count"],
        },
    }
    torch.save(payload, args.score_out)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trials": len(results["scores"]),
        "eer": float(results["eer"]),
        "min_dcf": float(results["min_dcf"]),
        "min_dcf_threshold": float(results["min_dcf_threshold"]),
        "score_file": str(args.score_out),
        "config": str(args.config),
        "checkpoint_path": str(args.checkpoint_path),
        "trials_jsonl": str(args.trials_jsonl),
        "device": str(device),
        "model_family": config.model.family,
        "side_pooling": family_side_pooling(config),
        "num_trials_total": results["num_trials_total"],
        "num_trials_scored": results["num_trials_scored"],
        "num_trials_skipped_bad": results["num_trials_skipped_bad"],
        "unique_side_count": results["unique_side_count"],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.tsv_out is not None:
        write_tsv(args.tsv_out, results["scores"], results["labels"])

    logging.info("Wrote score file: %s", args.score_out)
    logging.info("Wrote summary: %s", args.summary_out)
    if args.tsv_out is not None:
        logging.info("Wrote TSV: %s", args.tsv_out)


if __name__ == "__main__":
    main()
