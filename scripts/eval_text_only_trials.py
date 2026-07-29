"""Evaluate LUAR text encoders on text-only speaker-verification trials."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from multimodal_asv.evaluation.common import (
    compute_eer,
    compute_error_rates,
    compute_min_dcf,
    write_tsv,
)
from multimodal_asv.evaluation.text_only import (
    build_text_only_luar_encoder,
    encode_text_episode,
    extract_text_side,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for text-only LUAR trial evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LUAR text encoders directly on text-only trial JSONL files. "
            "Each side is encoded as one multi-utterance episode and scored with cosine similarity."
        )
    )
    parser.add_argument("--trials-jsonl", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="rrivera1849/LUAR-CRUD")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--score-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--tsv-out", type=Path, default=None)
    parser.add_argument("--skip-empty-transcript", action="store_true", default=True)
    parser.add_argument("--no-skip-empty-transcript", dest="skip_empty_transcript", action="store_false")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    """Score text-only trials, then write score tensors and summary metrics."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_text_only_luar_encoder(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint_path,
        hf_token=args.hf_token,
        device=device,
    )

    cache: Dict[Tuple[str, ...], torch.Tensor] = {}
    scores: List[float] = []
    labels: List[int] = []
    trials_total = 0
    trials_scored = 0
    skipped_empty = 0
    skipped_bad = 0

    with args.trials_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if args.max_trials > 0 and trials_total >= args.max_trials:
                break
            trials_total += 1
            trial = json.loads(line)

            enroll_texts = extract_text_side(trial.get("enroll") or {})
            test_texts = extract_text_side(trial.get("test") or {})
            if not enroll_texts or not test_texts:
                if args.skip_empty_transcript:
                    skipped_empty += 1
                    continue
                raise RuntimeError(f"Trial {trials_total} has empty transcripts on enroll/test side.")

            try:
                enroll_emb = encode_text_episode(encoder, enroll_texts, cache)
                test_emb = encode_text_episode(encoder, test_texts, cache)
            except Exception as exc:
                logging.warning("Skipping trial %d due to encoder failure: %s", trials_total, exc)
                skipped_bad += 1
                continue

            score = F.cosine_similarity(enroll_emb.reshape(-1), test_emb.reshape(-1), dim=0).item()
            scores.append(float(score))
            labels.append(int(trial.get("label", 0)))
            trials_scored += 1

            if args.log_every > 0 and trials_total % args.log_every == 0:
                logging.info(
                    "progress trials=%d scored=%d skipped_empty=%d skipped_bad=%d cache=%d",
                    trials_total,
                    trials_scored,
                    skipped_empty,
                    skipped_bad,
                    len(cache),
                )

    if not scores:
        raise RuntimeError("No trials scored. Check transcript coverage and encoder setup.")

    fnrs, fprs, thresholds = compute_error_rates(scores, labels)
    eer = compute_eer(fnrs, fprs)
    min_dcf, min_dcf_threshold = compute_min_dcf(fnrs, fprs, thresholds)

    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": scores,
        "labels": labels,
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trials_jsonl": str(args.trials_jsonl),
            "repo_id": args.repo_id,
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
            "device": device,
            "num_trials_total": trials_total,
            "num_trials_scored": trials_scored,
            "num_trials_skipped_empty": skipped_empty,
            "num_trials_skipped_bad": skipped_bad,
            "unique_episode_count": len(cache),
            "pooling": "episode_encoder_no_mean_pool",
        },
    }
    torch.save(payload, args.score_out)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trials": len(scores),
        "eer": float(eer),
        "min_dcf": float(min_dcf),
        "min_dcf_threshold": float(min_dcf_threshold),
        "score_file": str(args.score_out),
        "trials_jsonl": str(args.trials_jsonl),
        "repo_id": args.repo_id,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "device": device,
        "num_trials_total": trials_total,
        "num_trials_scored": trials_scored,
        "num_trials_skipped_empty": skipped_empty,
        "num_trials_skipped_bad": skipped_bad,
        "unique_episode_count": len(cache),
        "pooling": "episode_encoder_no_mean_pool",
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.tsv_out is not None:
        write_tsv(args.tsv_out, scores, labels)

    logging.info("Wrote score file: %s", args.score_out)
    logging.info("Wrote summary: %s", args.summary_out)
    if args.tsv_out is not None:
        logging.info("Wrote TSV: %s", args.tsv_out)


if __name__ == "__main__":
    main()
