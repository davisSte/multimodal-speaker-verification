"""Compute Local Information Disclosure and 1-to-N re-identification metrics.

This is a project-local implementation of the evaluation described by the Aalto
LID project: https://github.com/Speech-Interaction-Technology-Aalto-U/LID
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REQUIRED_COLUMNS = {"enroll_spk", "trial_spk", "trial_id", "score"}


@dataclass(frozen=True)
class Score:
    enroll_spk: str
    trial_spk: str
    trial_id: str
    score: float


@dataclass(frozen=True)
class Calibration:
    weight: float
    intercept: float
    n_scores: int
    n_mated: int
    n_non_mated: int


def read_scores(path: Path) -> list[Score]:
    """Read an exhaustive 1-to-N score matrix from CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: score must be finite") from exc
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: score must be finite")
            rows.append(
                Score(
                    enroll_spk=str(row["enroll_spk"]),
                    trial_spk=str(row["trial_spk"]),
                    trial_id=str(row["trial_id"]),
                    score=score,
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no scores")
    return rows


def group_and_validate(rows: Iterable[Score], *, name: str) -> dict[str, list[Score]]:
    """Validate the closed-set protocol and group scores by intercepted trial."""
    grouped: dict[str, list[Score]] = defaultdict(list)
    for row in rows:
        grouped[row.trial_id].append(row)

    expected_roster: frozenset[str] | None = None
    for trial_id, trial_rows in grouped.items():
        trial_speakers = {row.trial_spk for row in trial_rows}
        if len(trial_speakers) != 1:
            raise ValueError(f"{name}: trial {trial_id!r} has multiple trial_spk values")
        roster = [row.enroll_spk for row in trial_rows]
        if len(roster) != len(set(roster)):
            raise ValueError(f"{name}: trial {trial_id!r} has duplicate enrolment speakers")
        target = trial_rows[0].trial_spk
        if roster.count(target) != 1:
            raise ValueError(
                f"{name}: trial {trial_id!r} must have exactly one matching enrolment"
            )
        roster_set = frozenset(roster)
        if expected_roster is None:
            expected_roster = roster_set
        elif roster_set != expected_roster:
            raise ValueError(f"{name}: every trial must use the same enrolment roster")
        if len(trial_rows) < 2:
            raise ValueError(f"{name}: LID requires at least two enrolment candidates")
    return dict(grouped)


def z_normalize(grouped: dict[str, list[Score]]) -> dict[str, np.ndarray]:
    """Apply sample z-normalization independently to each 1-to-N score row."""
    normalized = {}
    for trial_id, rows in grouped.items():
        values = np.asarray([row.score for row in rows], dtype=np.float64)
        std = float(values.std(ddof=1))
        if not math.isfinite(std) or std == 0.0:
            raise ValueError(f"trial {trial_id!r} has zero score variance")
        normalized[trial_id] = (values - values.mean()) / std
    return normalized


def fit_calibration(
    grouped: dict[str, list[Score]], normalized: dict[str, np.ndarray]
) -> Calibration:
    """Fit the balanced linear logistic calibration used by LID on development data."""
    features = []
    labels = []
    for trial_id, rows in grouped.items():
        features.extend(normalized[trial_id].tolist())
        labels.extend(int(row.enroll_spk == row.trial_spk) for row in rows)
    x = np.asarray(features, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("development scores must contain mated and non-mated comparisons")
    model = LogisticRegression(class_weight="balanced")
    model.fit(x, y)
    return Calibration(
        weight=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        n_scores=len(y),
        n_mated=int(y.sum()),
        n_non_mated=int(len(y) - y.sum()),
    )


def _softmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = values - float(values.max())
    log_denominator = float(np.log(np.exp(shifted).sum()))
    log_probabilities = shifted - log_denominator
    return np.exp(log_probabilities), log_probabilities


def _expected_rank_metrics(probabilities: np.ndarray, target_index: int) -> tuple[float, float, float]:
    """Return tie-aware expected top-1, top-5, and reciprocal rank."""
    target = probabilities[target_index]
    ahead = int(np.count_nonzero(probabilities > target))
    tied = int(np.count_nonzero(np.isclose(probabilities, target, rtol=1e-12, atol=1e-15)))

    def expected_top_k(k: int) -> float:
        available = max(0, min(tied, k - ahead))
        return available / tied

    reciprocal_rank = sum(1.0 / rank for rank in range(ahead + 1, ahead + tied + 1)) / tied
    return expected_top_k(1), expected_top_k(min(5, len(probabilities))), reciprocal_rank


def evaluate(
    dev_rows: list[Score], test_rows: list[Score]
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Calibrate on development scores and evaluate LID on held-out test scores."""
    dev = group_and_validate(dev_rows, name="development scores")
    test = group_and_validate(test_rows, name="test scores")
    calibration = fit_calibration(dev, z_normalize(dev))
    test_z = z_normalize(test)

    trial_results: list[dict[str, object]] = []
    calibrated_rows: list[dict[str, object]] = []
    for trial_id, rows in test.items():
        llrs = calibration.weight * test_z[trial_id] + calibration.intercept
        probabilities, log_probabilities = _softmax(llrs)
        target_index = next(i for i, row in enumerate(rows) if row.enroll_spk == row.trial_spk)
        n_enrolments = len(rows)
        lid = math.log2(n_enrolments) + float(log_probabilities[target_index]) / math.log(2)
        top1, top5, reciprocal_rank = _expected_rank_metrics(probabilities, target_index)
        trial_results.append(
            {
                "trial_id": trial_id,
                "trial_spk": rows[target_index].trial_spk,
                "target_probability": float(probabilities[target_index]),
                "LID": lid,
                "top1_credit": top1,
                "top5_credit": top5,
                "reciprocal_rank": reciprocal_rank,
            }
        )
        for row, z_score, llr, probability, log_probability in zip(
            rows, test_z[trial_id], llrs, probabilities, log_probabilities
        ):
            calibrated_rows.append(
                {
                    "enroll_spk": row.enroll_spk,
                    "trial_spk": row.trial_spk,
                    "trial_id": row.trial_id,
                    "score": row.score,
                    "z_score": float(z_score),
                    "llr": float(llr),
                    "ln_p": float(log_probability),
                    "p": float(probability),
                }
            )

    lids = np.asarray([float(row["LID"]) for row in trial_results])
    positive = lids[lids > 0]
    negative = lids[lids <= 0]
    metrics: dict[str, object] = {
        "n_trials": len(trial_results),
        "n_enrolments": len(next(iter(test.values()))),
        "ALID": float(lids.mean()),
        "PDR": float((lids > 0).mean()),
        "NDR": float((lids <= 0).mean()),
        "LID+": float(positive.mean()) if len(positive) else None,
        "LID-": float(negative.mean()) if len(negative) else None,
        "LID_max": float(lids.max()),
        "rank_1_identification_rate": float(np.mean([row["top1_credit"] for row in trial_results])),
        "rank_5_identification_rate": float(np.mean([row["top5_credit"] for row in trial_results])),
        "mean_reciprocal_rank": float(np.mean([row["reciprocal_rank"] for row in trial_results])),
        "calibration": {
            "formula": "LLR = weight * per-trial-z-score + intercept",
            **calibration.__dict__,
        },
    }
    return metrics, trial_results, calibrated_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LID and closed-set 1-to-N re-identification metrics."
    )
    parser.add_argument("--dev-scores", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, trial_results, calibrated_rows = evaluate(
        read_scores(args.dev_scores), read_scores(args.test_scores)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_csv(args.output_dir / "local_information_disclosures.csv", trial_results)
    write_csv(args.output_dir / "calibrated_test_scores.csv", calibrated_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
