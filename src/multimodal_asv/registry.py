from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from .config import ExperimentConfig
from .models import (
    build_mean_pooled_audio_model,
    build_frame_pooled_audio_model,
    build_frame_pooled_audio_prosody_model,
    build_frame_pooled_audio_text_model,
    build_hybrid_audio_text_model,
    build_query_attention_audio_model,
    build_rjca_audio_global_luar_model,
    build_utterance_audio_prosody_model,
    build_utterance_audio_text_fusion_model,
    build_whisper_wavlm_cross_attention_model,
)


@dataclass(frozen=True)
class ModelFamily:
    """Metadata and builder entrypoint for one supported model family."""

    key: str
    display_name: str
    category: str
    description: str
    builder: Callable[[ExperimentConfig], object]


def build_not_implemented(config: ExperimentConfig) -> object:
    family = config.model.family
    raise NotImplementedError(
        f"Model family {family!r} is registered, but its builder has not been migrated yet."
    )


MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "mean_pooled_audio": ModelFamily(
        key="mean_pooled_audio",
        display_name="Mean-Pooled Audio Model",
        category="Audio Baseline",
        description="Averages audio utterance embeddings across multiple utterances.",
        builder=build_mean_pooled_audio_model,
    ),
    "query_attention_audio": ModelFamily(
        key="query_attention_audio",
        display_name="Query Attention Audio Model",
        category="Utterance-Level Aggregation",
        description="Aggregates audio utterance embeddings with a learned query.",
        builder=build_query_attention_audio_model,
    ),
    "utterance_audio_text_fusion": ModelFamily(
        key="utterance_audio_text_fusion",
        display_name="Utterance-Level Audio-Text Fusion",
        category="Utterance-Level Aggregation",
        description="Combines an audio utterance embedding with LUAR or prosody features.",
        builder=build_utterance_audio_text_fusion_model,
    ),
    "utterance_audio_prosody": ModelFamily(
        key="utterance_audio_prosody",
        display_name="Utterance-Level Audio-Prosody Fusion",
        category="Utterance-Level Aggregation",
        description="Combines an audio utterance embedding with pitch, speaking-rate, and voicing features.",
        builder=build_utterance_audio_prosody_model,
    ),
    "frame_pooled_audio": ModelFamily(
        key="frame_pooled_audio",
        display_name="Frame-Pooled Audio Model",
        category="Frame-Level Aggregation",
        description="Aggregates ECAPA frame-level audio representations across utterances.",
        builder=build_frame_pooled_audio_model,
    ),
    "frame_pooled_audio_prosody": ModelFamily(
        key="frame_pooled_audio_prosody",
        display_name="Frame-Pooled Audio-Prosody Fusion",
        category="Frame-Level Aggregation",
        description="Combines frame-level acoustic aggregation with grouped prosody statistics.",
        builder=build_frame_pooled_audio_prosody_model,
    ),
    "frame_pooled_audio_text": ModelFamily(
        key="frame_pooled_audio_text",
        display_name="Frame-Pooled Audio-Text Fusion",
        category="Frame-Level Aggregation",
        description="Combines frame-level acoustic aggregation with LUAR text features.",
        builder=build_frame_pooled_audio_text_model,
    ),
    "rjca_audio_global_luar": ModelFamily(
        key="rjca_audio_global_luar",
        display_name="RJCA Audio + Global LUAR",
        category="Cross-Representation Attention",
        description="Recursive joint cross-attention with grouped audio embeddings and one repeated LUAR embedding per speaker.",
        builder=build_rjca_audio_global_luar_model,
    ),
    "hybrid_audio_text": ModelFamily(
        key="hybrid_audio_text",
        display_name="Hybrid Audio-Text Cross-Attention",
        category="Cross-Representation Attention",
        description="Uses LUAR token embeddings to align transcript tokens against acoustic frames.",
        builder=build_hybrid_audio_text_model,
    ),
    "whisper_wavlm_cross_attention": ModelFamily(
        key="whisper_wavlm_cross_attention",
        display_name="Whisper-WavLM Cross-Attention Model",
        category="Cross-Representation Attention",
        description="Combines Whisper and WavLM sequence representations with cross-attention.",
        builder=build_whisper_wavlm_cross_attention_model,
    ),
}


def registered_model_keys() -> Iterable[str]:
    return MODEL_FAMILIES.keys()


def get_model_family(key: str) -> ModelFamily:
    try:
        return MODEL_FAMILIES[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_FAMILIES))
        raise KeyError(f"Unknown model family {key!r}. Known families: {known}") from exc


def build_model(config: ExperimentConfig) -> object:
    family = get_model_family(config.model.family)
    return family.builder(config)
