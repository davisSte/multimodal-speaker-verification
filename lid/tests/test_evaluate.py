from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("lid_evaluate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Score = MODULE.Score


def score_rows(correct_scores: list[float], incorrect_score: float = 0.0):
    speakers = [f"speaker_{index}" for index in range(len(correct_scores))]
    rows = []
    for trial_index, (trial_speaker, correct_score) in enumerate(
        zip(speakers, correct_scores), start=1
    ):
        for enroll_speaker in speakers:
            rows.append(
                Score(
                    enroll_spk=enroll_speaker,
                    trial_spk=trial_speaker,
                    trial_id=f"trial_{trial_index}",
                    score=correct_score if enroll_speaker == trial_speaker else incorrect_score,
                )
            )
    return rows


def test_evaluate_reports_lid_and_identification_metrics():
    dev = score_rows([0.9, 0.8, 0.7])
    test = score_rows([0.95, 0.85, 0.75])

    metrics, trials, calibrated = MODULE.evaluate(dev, test)

    assert metrics["n_trials"] == 3
    assert metrics["n_enrolments"] == 3
    assert metrics["rank_1_identification_rate"] == pytest.approx(1.0)
    assert metrics["mean_reciprocal_rank"] == pytest.approx(1.0)
    assert metrics["ALID"] > 0
    assert metrics["PDR"] == pytest.approx(1.0)
    assert len(trials) == 3
    assert len(calibrated) == 9
    for trial_id in {row["trial_id"] for row in calibrated}:
        total = sum(row["p"] for row in calibrated if row["trial_id"] == trial_id)
        assert total == pytest.approx(1.0)


def test_rejects_non_exhaustive_gallery():
    rows = score_rows([0.9, 0.8, 0.7])
    rows.pop()

    with pytest.raises(ValueError, match="matching enrolment|same enrolment roster"):
        MODULE.group_and_validate(rows, name="scores")


def test_ties_receive_expected_random_tie_break_credit():
    probabilities = MODULE.np.asarray([0.4, 0.4, 0.2])

    top1, top5, reciprocal_rank = MODULE._expected_rank_metrics(probabilities, 0)

    assert top1 == pytest.approx(0.5)
    assert top5 == pytest.approx(1.0)
    assert reciprocal_rank == pytest.approx(0.75)
