from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.registry import build_model
from multimodal_asv.utils.prosody import (
    extract_prosody_feature_tensor,
    get_word_timestamps_for_audio,
    load_whisper_transcript_index,
)
from .common import compute_eer, compute_error_rates, compute_min_dcf, extract_texts_from_side


GROUPED_AUDIO_FAMILIES = {
    "mean_pooled_audio",
    "query_attention_audio",
    "frame_pooled_audio",
    "whisper_wavlm_cross_attention",
}
GROUPED_AUDIO_TEXT_FAMILIES = {
    "frame_pooled_audio_text",
    "rjca_audio_global_luar",
}
GROUPED_AUDIO_PROSODY_FAMILIES = {
    "frame_pooled_audio_prosody",
}
FLAT_UTTERANCE_FAMILIES = {
    "utterance_audio_text_fusion",
    "hybrid_audio_text",
}


def extract_audio_paths_from_side(side: Dict[str, Any], *, trial_dir: Path) -> List[Path]:
    items = side.get("audio_paths", side.get("paths", []))
    if not isinstance(items, list):
        return []
    out: List[Path] = []
    for item in items:
        path = Path(str(item))
        if not path.is_absolute():
            path = (trial_dir / path).resolve()
        out.append(path)
    return out


def load_waveforms(
    paths: Sequence[Path],
    *,
    sample_rate: int,
    segment_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_samples = int(round(sample_rate * segment_seconds))
    if target_samples <= 0:
        raise ValueError("segment_seconds must be positive.")

    waveforms: List[torch.Tensor] = []
    lengths: List[int] = []
    for path in paths:
        waveform, source_rate = torchaudio.load(str(path))
        waveform = _to_mono(waveform)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                source_rate,
                sample_rate,
            ).squeeze(0)
        waveform, valid_length = _crop_or_pad_center(waveform, target_samples)
        waveforms.append(waveform)
        lengths.append(valid_length)
    return torch.stack(waveforms, dim=0), torch.tensor(lengths, dtype=torch.long)


def load_checkpoint_model(config: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config)
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def family_side_pooling(config: ExperimentConfig) -> str:
    family = config.model.family
    if family in GROUPED_AUDIO_FAMILIES or family in GROUPED_AUDIO_TEXT_FAMILIES or family in GROUPED_AUDIO_PROSODY_FAMILIES:
        return "model_grouped_episode"
    return "mean_pool_utterance_embeddings"


def utterance_fusion_modalities(config: ExperimentConfig) -> tuple[str, ...]:
    fusion = dict(config.model.options.get("fusion", {}) or {})
    modalities = fusion.get("modalities", ("audio", "text"))
    return tuple(str(item) for item in modalities)


def canonical_side_key(
    config: ExperimentConfig,
    side: Dict[str, Any],
    *,
    trial_dir: Path,
) -> Tuple[Any, ...]:
    paths = tuple(str(path) for path in extract_audio_paths_from_side(side, trial_dir=trial_dir))
    transcripts = tuple(extract_texts_from_side(side))
    return (config.model.family, paths, transcripts)


def maybe_load_prosody_index(config: ExperimentConfig) -> dict[str, dict[str, Any]] | None:
    if (
        config.model.family in GROUPED_AUDIO_PROSODY_FAMILIES
        or (
            config.model.family in FLAT_UTTERANCE_FAMILIES
            and "pitch" in utterance_fusion_modalities(config)
            and "text" not in utterance_fusion_modalities(config)
        )
    ):
        transcript_path = config.data.options.get("prosody_transcript_path")
        if transcript_path is None:
            raise RuntimeError(
                f"{config.model.family} evaluation requires data.prosody_transcript_path in the config."
            )
        return load_whisper_transcript_index(transcript_path)
    return None


def encode_trial_side(
    model: torch.nn.Module,
    config: ExperimentConfig,
    side: Dict[str, Any],
    *,
    trial_dir: Path,
    transcript_index: dict[str, dict[str, Any]] | None,
    device: torch.device,
) -> torch.Tensor:
    paths = extract_audio_paths_from_side(side, trial_dir=trial_dir)
    transcripts = extract_texts_from_side(side)
    family = config.model.family

    if family in GROUPED_AUDIO_FAMILIES:
        if not paths:
            raise RuntimeError(f"{family} requires side.audio_paths.")
        waveforms, lengths = load_waveforms(
            paths,
            sample_rate=config.data.sample_rate,
            segment_seconds=config.data.segment_seconds,
        )
        waveforms = waveforms.unsqueeze(0).to(device)
        lengths = lengths.unsqueeze(0)
        padding_mask = make_padding_mask(lengths.view(-1), waveforms.size(-1), device).view(1, waveforms.size(1), waveforms.size(2))
        with torch.no_grad():
            emb = model(
                waveforms=waveforms,
                padding_mask=padding_mask,
                num_speakers=1,
                utterances_per_speaker=waveforms.size(1),
            )
        return as_normalized_vector(emb)

    if family in GROUPED_AUDIO_PROSODY_FAMILIES:
        if not paths:
            raise RuntimeError(f"{family} requires side.audio_paths.")
        waveforms, lengths = load_waveforms(
            paths,
            sample_rate=config.data.sample_rate,
            segment_seconds=config.data.segment_seconds,
        )
        prosody_feature_set = str(config.data.options.get("prosody_feature_set", "pitch_rate_voicing"))
        prosody_rows = []
        for path, waveform in zip(paths, waveforms):
            word_timestamps = get_word_timestamps_for_audio(path, transcript_index=transcript_index)
            prosody_rows.append(
                extract_prosody_feature_tensor(
                    waveform,
                    config.data.sample_rate,
                    feature_set=prosody_feature_set,
                    word_timestamps=word_timestamps,
                )
            )
        pitch_features_grouped = torch.stack(prosody_rows, dim=0).unsqueeze(0).to(device)
        waveforms = waveforms.unsqueeze(0).to(device)
        lengths = lengths.unsqueeze(0)
        padding_mask = make_padding_mask(lengths.view(-1), waveforms.size(-1), device).view(1, waveforms.size(1), waveforms.size(2))
        with torch.no_grad():
            emb = model(
                waveforms=waveforms,
                padding_mask=padding_mask,
                pitch_features_grouped=pitch_features_grouped,
                num_speakers=1,
                utterances_per_speaker=waveforms.size(1),
            )
        return as_normalized_vector(emb)

    if family in GROUPED_AUDIO_TEXT_FAMILIES:
        if not paths:
            raise RuntimeError(f"{family} requires side.audio_paths.")
        if not transcripts:
            raise RuntimeError(f"{family} requires side.transcripts.")
        if len(paths) != len(transcripts):
            raise RuntimeError(f"{family} requires matching numbers of audio paths and transcripts.")
        waveforms, lengths = load_waveforms(
            paths,
            sample_rate=config.data.sample_rate,
            segment_seconds=config.data.segment_seconds,
        )
        waveforms = waveforms.unsqueeze(0).to(device)
        lengths = lengths.unsqueeze(0)
        padding_mask = make_padding_mask(lengths.view(-1), waveforms.size(-1), device).view(1, waveforms.size(1), waveforms.size(2))
        with torch.no_grad():
            emb = model(
                waveforms=waveforms,
                transcripts_grouped=[transcripts],
                padding_mask=padding_mask,
                num_speakers=1,
                utterances_per_speaker=waveforms.size(1),
            )
        return as_normalized_vector(emb)

    if family in FLAT_UTTERANCE_FAMILIES:
        if not paths:
            raise RuntimeError(f"{family} requires side.audio_paths.")
        waveforms, lengths = load_waveforms(
            paths,
            sample_rate=config.data.sample_rate,
            segment_seconds=config.data.segment_seconds,
        )
        padding_mask = make_padding_mask(lengths, waveforms.size(1), device)
        waveforms = waveforms.to(device)
        modalities = utterance_fusion_modalities(config)
        with torch.no_grad():
            if family == "hybrid_audio_text":
                if not transcripts or len(paths) != len(transcripts):
                    raise RuntimeError("hybrid_audio_text requires one transcript per audio path.")
                emb = model(waveforms=waveforms, transcripts=transcripts, padding_mask=padding_mask)
            elif "pitch" in modalities and "text" not in modalities:
                prosody_feature_set = str(config.data.options.get("prosody_feature_set", "pitch_rate_voicing"))
                pitch_features = []
                for path, waveform in zip(paths, waveforms.cpu()):
                    word_timestamps = get_word_timestamps_for_audio(path, transcript_index=transcript_index)
                    pitch_features.append(
                        extract_prosody_feature_tensor(
                            waveform,
                            config.data.sample_rate,
                            feature_set=prosody_feature_set,
                            word_timestamps=word_timestamps,
                        )
                    )
                emb = model(
                    waveforms=waveforms,
                    padding_mask=padding_mask,
                    pitch_features=torch.stack(pitch_features, dim=0).to(device),
                )
            else:
                if not transcripts or len(paths) != len(transcripts):
                    raise RuntimeError(f"{family} requires one transcript per audio path.")
                emb = model(waveforms=waveforms, transcripts=transcripts, padding_mask=padding_mask)
        emb = emb.mean(dim=0, keepdim=True)
        return as_normalized_vector(emb)

    raise RuntimeError(f"Unsupported family for checkpoint trial evaluation: {family}")


def evaluate_checkpoint_trials(
    model: torch.nn.Module,
    config: ExperimentConfig,
    *,
    trials_jsonl: Path,
    device: torch.device,
    max_trials: int = 0,
    log_every: int = 500,
) -> dict[str, Any]:
    transcript_index = maybe_load_prosody_index(config)
    cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
    scores: List[float] = []
    labels: List[int] = []
    trials_total = 0
    trials_scored = 0
    skipped_bad = 0
    trial_dir = trials_jsonl.parent.resolve()

    with trials_jsonl.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if max_trials > 0 and trials_total >= max_trials:
                break
            trials_total += 1
            trial = json.loads(line)
            enroll = trial.get("enroll") or {}
            test = trial.get("test") or {}
            if not isinstance(enroll, dict) or not isinstance(test, dict):
                raise RuntimeError(f"Trial {trials_total} must contain enroll/test objects.")

            try:
                enroll_key = canonical_side_key(config, enroll, trial_dir=trial_dir)
                test_key = canonical_side_key(config, test, trial_dir=trial_dir)
                enroll_emb = cache.get(enroll_key)
                if enroll_emb is None:
                    enroll_emb = encode_trial_side(
                        model,
                        config,
                        enroll,
                        trial_dir=trial_dir,
                        transcript_index=transcript_index,
                        device=device,
                    )
                    cache[enroll_key] = enroll_emb
                test_emb = cache.get(test_key)
                if test_emb is None:
                    test_emb = encode_trial_side(
                        model,
                        config,
                        test,
                        trial_dir=trial_dir,
                        transcript_index=transcript_index,
                        device=device,
                    )
                    cache[test_key] = test_emb
            except Exception as exc:
                logging.warning("Skipping trial %d due to scoring failure: %s", trials_total, exc)
                skipped_bad += 1
                continue

            score = F.cosine_similarity(enroll_emb.reshape(-1), test_emb.reshape(-1), dim=0).item()
            scores.append(float(score))
            labels.append(int(trial.get("label", 0)))
            trials_scored += 1

            if log_every > 0 and trials_total % log_every == 0:
                logging.info(
                    "progress trials=%d scored=%d skipped_bad=%d cache=%d",
                    trials_total,
                    trials_scored,
                    skipped_bad,
                    len(cache),
                )

    if not scores:
        raise RuntimeError("No trials scored. Check the trial format, config, and checkpoint.")

    fnrs, fprs, thresholds = compute_error_rates(scores, labels)
    return {
        "scores": scores,
        "labels": labels,
        "eer": compute_eer(fnrs, fprs),
        "min_dcf": compute_min_dcf(fnrs, fprs, thresholds)[0],
        "min_dcf_threshold": compute_min_dcf(fnrs, fprs, thresholds)[1],
        "num_trials_total": trials_total,
        "num_trials_scored": trials_scored,
        "num_trials_skipped_bad": skipped_bad,
        "unique_side_count": len(cache),
        "side_pooling": family_side_pooling(config),
    }


def as_normalized_vector(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2 or embedding.size(0) != 1:
        raise RuntimeError(f"Expected one side embedding with shape (1, dim), got {tuple(embedding.shape)}")
    return F.normalize(embedding[0].to(torch.float32).detach().cpu(), dim=0)


def make_padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    rng = torch.arange(max_len, device=device).unsqueeze(0)
    return rng >= lengths.to(device).unsqueeze(1)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    if waveform.numel() == 0:
        return torch.zeros(1, dtype=torch.float32)
    return waveform.to(torch.float32)


def _crop_or_pad_center(waveform: torch.Tensor, target_samples: int) -> tuple[torch.Tensor, int]:
    num_samples = waveform.size(-1)
    valid_length = min(num_samples, target_samples)
    if num_samples > target_samples:
        start = (num_samples - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    elif num_samples < target_samples:
        waveform = F.pad(waveform, (0, target_samples - num_samples))
    return waveform.contiguous(), valid_length
