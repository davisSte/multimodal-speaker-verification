#!/usr/bin/env python3
"""Evaluate trained audio or multimodal checkpoints on trial JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch

from multimodal_asv.config import load_config
from multimodal_asv.evaluation.common import write_tsv
from multimodal_asv.evaluation.trials import (
    evaluate_checkpoint_trials,
    family_side_pooling,
    load_checkpoint_model,
)
from multimodal_asv.utils.huggingface import resolve_hf_or_local_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint-based trial evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate audio and multimodal ASV checkpoints on trial JSONL files. "
            "The model is rebuilt from a config and restored from a checkpoint."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment config YAML, or hf://owner/repo/path/to/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Training checkpoint path, or hf://owner/repo/path/to/checkpoint.pt.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Optional Hugging Face download cache.")
    parser.add_argument("--wavlm-checkpoint", type=Path, default=None, help="Override model.audio_encoder.wavlm_checkpoint.")
    parser.add_argument(
        "--text-encoder-checkpoint",
        type=Path,
        default=None,
        help="Override model.text_encoder.checkpoint_path for released audio-text checkpoints.",
    )
    parser.add_argument("--trials-jsonl", type=Path, required=True, help="Trial JSONL with enroll/test sides.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--score-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--tsv-out", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    """Restore a checkpoint, score trials, and write summary artifacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = resolve_hf_or_local_path(args.config, cache_dir=args.hf_cache_dir)
    checkpoint_path = resolve_hf_or_local_path(args.checkpoint_path, cache_dir=args.hf_cache_dir)

    config = load_config(config_path)
    apply_runtime_overrides(config, args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint_model(config, checkpoint_path, device)
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
            "resolved_config": str(config_path),
            "checkpoint_path": str(args.checkpoint_path),
            "resolved_checkpoint_path": str(checkpoint_path),
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
        "resolved_config": str(config_path),
        "checkpoint_path": str(args.checkpoint_path),
        "resolved_checkpoint_path": str(checkpoint_path),
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


def apply_runtime_overrides(config, args: argparse.Namespace) -> None:
    """Apply local asset-path overrides to a loaded experiment config."""

    if args.wavlm_checkpoint is not None:
        audio_encoder = config.model.options.get("audio_encoder")
        if not isinstance(audio_encoder, dict):
            raise ValueError("--wavlm-checkpoint requires model.audio_encoder in the config.")
        audio_encoder["wavlm_checkpoint"] = str(args.wavlm_checkpoint)

    if args.text_encoder_checkpoint is not None:
        text_encoder = config.model.options.get("text_encoder")
        if not isinstance(text_encoder, dict):
            raise ValueError("--text-encoder-checkpoint requires model.text_encoder in the config.")
        text_encoder["checkpoint_path"] = str(args.text_encoder_checkpoint)


if __name__ == "__main__":
    main()
