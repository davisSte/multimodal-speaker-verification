from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import parselmouth
import torch

try:
    import pronouncing
except Exception:
    pronouncing = None


PROSODY_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "pitch_rate_voicing": ("F0_mean", "speaking_rate", "voiced_ratio"),
}

_WHISPER_INDEX_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def get_prosody_feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set not in PROSODY_FEATURE_SETS:
        raise ValueError(f"Unsupported prosody feature set: {feature_set!r}")
    return PROSODY_FEATURE_SETS[feature_set]


def get_prosody_feature_dim(feature_set: str) -> int:
    return len(get_prosody_feature_names(feature_set))


def extract_prosody_feature_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    feature_set: str,
    word_timestamps: Sequence[dict[str, Any]] | None = None,
) -> torch.Tensor:
    if feature_set != "pitch_rate_voicing":
        raise ValueError(f"Unsupported prosody feature set: {feature_set!r}")
    features = extract_pitch_rate_voicing_features(
        waveform,
        sample_rate,
        word_timestamps=word_timestamps,
    )
    return torch.tensor(
        [float(features[name]) for name in get_prosody_feature_names(feature_set)],
        dtype=torch.float32,
    )


def extract_pitch_rate_voicing_features(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    word_timestamps: Sequence[dict[str, Any]] | None = None,
    pitch_floor: float = 50.0,
    pitch_ceiling: float = 400.0,
    time_step: float = 0.01,
) -> dict[str, float]:
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform tensor, got shape {tuple(waveform.shape)}")
    if waveform.numel() == 0:
        return {"F0_mean": 0.0, "speaking_rate": 0.0, "voiced_ratio": 0.0}

    samples = waveform.detach().cpu().to(torch.float32).numpy()
    snd = parselmouth.Sound(samples, sampling_frequency=float(sample_rate))
    pitch = snd.to_pitch(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    f0_all = pitch.selected_array["frequency"]
    if f0_all.size == 0:
        return {"F0_mean": 0.0, "speaking_rate": 0.0, "voiced_ratio": 0.0}

    f0 = f0_all[f0_all > 0]
    if f0.size == 0:
        f0_mean = 0.0
        voiced_ratio = 0.0
    else:
        f0_mean = float(np.mean(f0))
        voiced_ratio = float(f0.size) / float(f0_all.size)

    speaking_rate = _estimate_speaking_rate(word_timestamps or [])
    return {
        "F0_mean": f0_mean,
        "speaking_rate": speaking_rate,
        "voiced_ratio": voiced_ratio,
    }


def _estimate_speaking_rate(word_timestamps: Sequence[dict[str, Any]]) -> float:
    words = _coerce_word_timestamps(word_timestamps)
    if len(words) < 2:
        return 0.0
    total_syllables = sum(count_syllables(str(word["word"])) for word in words)
    duration = max(0.0, float(words[-1]["end"]) - float(words[0]["start"]))
    if duration <= 0.0:
        return 0.0
    return float(total_syllables / duration)


def load_whisper_transcript_index(path_or_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(path_or_dir).expanduser().resolve()
    cache_key = str(root)
    cached = _WHISPER_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if root.is_dir():
        files = sorted(root.glob("data_*.whisper_medium.en.jsonl"))
        if not files:
            files = sorted(root.glob("*.jsonl"))
    else:
        files = [root]
    if not files:
        raise FileNotFoundError(f"No Whisper transcript JSONL files found under {root}")

    index: dict[str, dict[str, Any]] = {}
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                basename = Path(str(record.get("file", ""))).name
                if not basename:
                    continue
                words: list[dict[str, Any]] = []
                for seg in record.get("segments", []):
                    if not isinstance(seg, dict):
                        continue
                    for word in seg.get("words", []):
                        if not isinstance(word, dict):
                            continue
                        if "word" not in word or "start" not in word or "end" not in word:
                            continue
                        words.append(
                            {
                                "word": str(word["word"]),
                                "start": float(word["start"]),
                                "end": float(word["end"]),
                            }
                        )
                index[basename] = {"words": words}

    _WHISPER_INDEX_CACHE[cache_key] = index
    return index


def get_word_timestamps_for_audio(
    audio_path: str | Path,
    *,
    transcript_index: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if transcript_index is None:
        return []
    basename = Path(audio_path).name
    record = transcript_index.get(basename)
    if record is None:
        return []
    words = record.get("words")
    if not isinstance(words, list):
        return []
    return _coerce_word_timestamps(words)


def _coerce_word_timestamps(word_timestamps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for item in word_timestamps:
        if not isinstance(item, dict):
            continue
        if "word" not in item or "start" not in item or "end" not in item:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except Exception:
            continue
        words.append({"word": str(item["word"]), "start": start, "end": end})
    return words


def count_syllables(word: str) -> int:
    cleaned = _clean_word_token(word)
    if not cleaned:
        return 0
    if pronouncing is not None:
        phones = pronouncing.phones_for_word(cleaned)
        if phones:
            return int(pronouncing.syllable_count(phones[0]))
    return _heuristic_syllable_count(cleaned)


def _clean_word_token(token: str) -> str:
    token = str(token).strip().lower()
    token = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", token)
    return token


def _heuristic_syllable_count(word: str) -> int:
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


__all__ = [
    "extract_prosody_feature_tensor",
    "get_prosody_feature_dim",
    "get_prosody_feature_names",
    "get_word_timestamps_for_audio",
    "load_whisper_transcript_index",
]
