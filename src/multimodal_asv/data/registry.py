from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from torch.utils.data import DataLoader, Dataset

from multimodal_asv.config import ExperimentConfig

from .audio import (
    build_flat_audio_prosody_dataloader,
    build_flat_audio_text_dataloader,
    build_grouped_audio_dataloader,
    build_grouped_audio_prosody_dataloader,
    build_grouped_audio_text_dataloader,
)


@dataclass(frozen=True)
class DataPipeline:
    """Dataset and dataloader pair built from an experiment config."""

    dataset: Dataset
    dataloader: DataLoader


DataBuilder = Callable[[ExperimentConfig], DataPipeline]


def build_grouped_audio_family_data(config: ExperimentConfig) -> DataPipeline:
    if config.data.manifest is None:
        raise ValueError(f"{config.model.family} requires data.manifest.")
    options = dict(config.data.options)
    dataset, dataloader = build_grouped_audio_dataloader(
        manifest_path=config.data.manifest,
        speakers_per_batch=int(options.pop("speakers_per_batch")),
        utterances_per_speaker=int(options.pop("utterances_per_speaker")),
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_segment=bool(options.pop("random_segment", True)),
        num_workers=int(options.pop("num_workers", 4)),
        seed=_optional_int(options.pop("seed", None)),
    )
    return DataPipeline(dataset=dataset, dataloader=dataloader)


def build_grouped_audio_text_family_data(config: ExperimentConfig) -> DataPipeline:
    if config.data.manifest is None:
        raise ValueError(f"{config.model.family} requires data.manifest.")
    options = dict(config.data.options)
    dataset, dataloader = build_grouped_audio_text_dataloader(
        manifest_path=config.data.manifest,
        speakers_per_batch=int(options.pop("speakers_per_batch")),
        utterances_per_speaker=int(options.pop("utterances_per_speaker")),
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_segment=bool(options.pop("random_segment", True)),
        num_workers=int(options.pop("num_workers", 4)),
        seed=_optional_int(options.pop("seed", None)),
    )
    return DataPipeline(dataset=dataset, dataloader=dataloader)


def build_grouped_audio_prosody_family_data(config: ExperimentConfig) -> DataPipeline:
    if config.data.manifest is None:
        raise ValueError(f"{config.model.family} requires data.manifest.")
    options = dict(config.data.options)
    dataset, dataloader = build_grouped_audio_prosody_dataloader(
        manifest_path=config.data.manifest,
        speakers_per_batch=int(options.pop("speakers_per_batch")),
        utterances_per_speaker=int(options.pop("utterances_per_speaker")),
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_segment=bool(options.pop("random_segment", True)),
        num_workers=int(options.pop("num_workers", 4)),
        seed=_optional_int(options.pop("seed", None)),
        prosody_feature_set=str(options.pop("prosody_feature_set", "pitch_rate_voicing")),
        prosody_transcript_path=options.pop("prosody_transcript_path", None),
    )
    return DataPipeline(dataset=dataset, dataloader=dataloader)


def build_flat_audio_text_family_data(config: ExperimentConfig) -> DataPipeline:
    if config.data.manifest is None:
        raise ValueError(f"{config.model.family} requires data.manifest.")
    options = dict(config.data.options)
    dataset, dataloader = build_flat_audio_text_dataloader(
        manifest_path=config.data.manifest,
        batch_size=config.training.batch_size,
        sample_rate=config.data.sample_rate,
        segment_seconds=config.data.segment_seconds,
        random_segment=bool(options.pop("random_segment", True)),
        num_workers=int(options.pop("num_workers", 4)),
        seed=_optional_int(options.pop("seed", None)),
        shuffle=bool(options.pop("shuffle", True)),
    )
    return DataPipeline(dataset=dataset, dataloader=dataloader)


def build_utterance_fusion_family_data(config: ExperimentConfig) -> DataPipeline:
    if config.data.manifest is None:
        raise ValueError(f"{config.model.family} requires data.manifest.")
    options = dict(config.data.options)
    model_options = dict(config.model.options)
    fusion = dict(model_options.get("fusion", {}) or {})
    modalities = ("audio", "pitch") if config.model.family == "utterance_audio_prosody" else tuple(fusion.get("modalities", ("audio", "text")))
    if "pitch" in modalities and "text" not in modalities:
        dataset, dataloader = build_flat_audio_prosody_dataloader(
            manifest_path=config.data.manifest,
            batch_size=config.training.batch_size,
            sample_rate=config.data.sample_rate,
            segment_seconds=config.data.segment_seconds,
            random_segment=bool(options.pop("random_segment", True)),
            num_workers=int(options.pop("num_workers", 4)),
            seed=_optional_int(options.pop("seed", None)),
            shuffle=bool(options.pop("shuffle", True)),
            prosody_feature_set=str(options.pop("prosody_feature_set", "pitch_rate_voicing")),
            prosody_transcript_path=options.pop("prosody_transcript_path", None),
        )
        return DataPipeline(dataset=dataset, dataloader=dataloader)
    return build_flat_audio_text_family_data(config)


DATA_BUILDERS: Dict[str, DataBuilder] = {
    "frame_pooled_audio": build_grouped_audio_family_data,
    "frame_pooled_audio_prosody": build_grouped_audio_prosody_family_data,
    "frame_pooled_audio_text": build_grouped_audio_text_family_data,
    "mean_pooled_audio": build_grouped_audio_family_data,
    "hybrid_audio_text": build_flat_audio_text_family_data,
    "query_attention_audio": build_grouped_audio_family_data,
    "rjca_audio_global_luar": build_grouped_audio_text_family_data,
    "utterance_audio_prosody": build_utterance_fusion_family_data,
    "utterance_audio_text_fusion": build_utterance_fusion_family_data,
    "whisper_wavlm_cross_attention": build_grouped_audio_family_data,
}


def build_data_pipeline(config: ExperimentConfig) -> DataPipeline:
    try:
        builder = DATA_BUILDERS[config.model.family]
    except KeyError as exc:
        known = ", ".join(sorted(DATA_BUILDERS))
        raise KeyError(
            f"No data builder registered for model family {config.model.family!r}. "
            f"Known data builders: {known}"
        ) from exc
    return builder(config)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = ["DataPipeline", "build_data_pipeline", "build_grouped_audio_family_data"]
