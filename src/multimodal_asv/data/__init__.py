"""Data loading utilities for multimodal ASV experiments."""

from .audio import (
    AudioTextUtteranceDataset,
    AudioUtteranceDataset,
    GroupedSpeakerBatchSampler,
    build_flat_audio_text_dataloader,
    build_grouped_audio_dataloader,
    build_grouped_audio_text_dataloader,
    load_audio,
    pad_audio_batch,
    pad_audio_text_batch,
    pad_grouped_audio_batch,
    pad_grouped_audio_text_batch,
    read_audio_manifest,
    read_audio_text_manifest,
)
from .registry import DataPipeline, build_data_pipeline

__all__ = [
    "AudioTextUtteranceDataset",
    "AudioUtteranceDataset",
    "DataPipeline",
    "GroupedSpeakerBatchSampler",
    "build_data_pipeline",
    "build_flat_audio_text_dataloader",
    "build_grouped_audio_dataloader",
    "build_grouped_audio_text_dataloader",
    "load_audio",
    "pad_audio_batch",
    "pad_audio_text_batch",
    "pad_grouped_audio_batch",
    "pad_grouped_audio_text_batch",
    "read_audio_manifest",
    "read_audio_text_manifest",
]
