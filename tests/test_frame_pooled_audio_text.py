from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import StaticTextEncoder
from multimodal_asv.models import (
    FramePooledAudioModel,
    FramePooledAudioTextModel,
    MeanStdFramePooler,
    ProjectedConcatFusion,
    WeightedSumFusion,
)
from multimodal_asv.registry import build_model


def test_frame_pooled_audio_text_projected_concat_shape() -> None:
    audio_model = FramePooledAudioModel(
        frame_pooler=MeanStdFramePooler(frame_dim=4, embedding_dim=6),
        embedding_dim=6,
    )
    model = FramePooledAudioTextModel(
        audio_model=audio_model,
        text_encoder=StaticTextEncoder(embedding_dim=8),
        fusion=ProjectedConcatFusion(
            input_dims={"audio": 6, "text": 8},
            projection_dim=10,
            final_dim=5,
        ),
    )

    fused = model(
        frame_features=torch.randn(3, 5, 4, 7),
        transcripts_grouped=[
            ["hello", "speaker"],
            ["another", "speaker"],
            ["third", "voice"],
        ],
    )

    assert fused.shape == (3, 5)
    assert torch.allclose(fused.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_frame_pooled_audio_text_weighted_sum_shape_and_stats() -> None:
    audio_model = FramePooledAudioModel(
        frame_pooler=MeanStdFramePooler(frame_dim=4, embedding_dim=6),
        embedding_dim=6,
    )
    fusion = WeightedSumFusion(
        audio_dim=6,
        text_dim=8,
        final_dim=5,
        audio_weight=0.2,
        text_weight=0.8,
        normalize_branches=True,
    )
    model = FramePooledAudioTextModel(
        audio_model=audio_model,
        text_encoder=StaticTextEncoder(embedding_dim=8),
        fusion=fusion,
    )

    fused = model(
        frame_features=torch.randn(2, 5, 4, 7),
        transcripts_grouped=[["hello"], ["speaker"]],
    )

    assert fused.shape == (2, 5)
    assert fusion.last_fusion_stats["audio_weight"] == 0.2
    assert fusion.last_fusion_stats["text_weight"] == 0.8


def test_frame_pooled_audio_text_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_frame_pooled_audio_text
model:
  family: frame_pooled_audio_text
  text_encoder:
    type: static
    embedding_dim: 8
  audio:
    embedding_dim: 6
  frame_pooler:
    type: mean_std
    frame_dim: 4
  fusion:
    type: weighted_sum
    audio_dim: 6
    text_dim: 8
    final_dim: 5
    audio_weight: 0.5
    text_weight: 0.5
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, FramePooledAudioTextModel)
    assert isinstance(model.text_encoder, StaticTextEncoder)
    assert isinstance(model.fusion, WeightedSumFusion)
