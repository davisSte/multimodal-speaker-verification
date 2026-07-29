from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence


def clean_texts(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def extract_texts_from_side(side: dict[str, Any]) -> List[str]:
    transcripts = side.get("transcripts")
    if not isinstance(transcripts, list):
        return []
    return clean_texts(transcripts)


def write_tsv(path: Path, scores: List[float], labels: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("index\tscore\tlabel\n")
        for index, (score, label) in enumerate(zip(scores, labels)):
            handle.write(f"{index}\t{score:.10f}\t{label}\n")


def compute_error_rates(scores: Sequence[float], labels: Sequence[int]) -> tuple[List[float], List[float], List[float]]:
    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        raise RuntimeError("EER/minDCF require both positive and negative trials.")

    fn = 0
    tn = 0
    fnrs: List[float] = []
    fprs: List[float] = []
    thresholds: List[float] = []
    for score, label in paired:
        if int(label) == 1:
            fn += 1
        else:
            tn += 1
        fnrs.append(fn / positives)
        fprs.append((negatives - tn) / negatives)
        thresholds.append(float(score))
    return fnrs, fprs, thresholds


def compute_eer(fnrs: Sequence[float], fprs: Sequence[float]) -> float:
    best_index = min(range(len(fnrs)), key=lambda idx: abs(fnrs[idx] - fprs[idx]))
    return float((fnrs[best_index] + fprs[best_index]) / 2.0)


def compute_min_dcf(
    fnrs: Sequence[float],
    fprs: Sequence[float],
    thresholds: Sequence[float],
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    best_cost = float("inf")
    best_threshold = float(thresholds[0]) if thresholds else 0.0
    default_cost = min(p_target * c_miss, (1.0 - p_target) * c_fa)
    for fnr, fpr, threshold in zip(fnrs, fprs, thresholds):
        cost = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)
    return float(best_cost / default_cost), best_threshold
