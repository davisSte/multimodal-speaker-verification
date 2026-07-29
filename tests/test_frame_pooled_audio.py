from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.models import FramePooledAudioModel, MeanStdFramePooler
from multimodal_asv.registry import build_model


def test_frame_pooled_audio_concatenates_utterance_frames() -> None:
    pooler = MeanStdFramePooler(frame_dim=4, embedding_dim=6)
    model = FramePooledAudioModel(frame_pooler=pooler, embedding_dim=6)
    frame_features = torch.randn(3, 5, 4, 7)

    speaker_embeddings = model(frame_features=frame_features)

    assert speaker_embeddings.shape == (3, 6)
    assert torch.allclose(speaker_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_frame_pooled_audio_groups_flat_frame_maps() -> None:
    pooler = MeanStdFramePooler(frame_dim=4, embedding_dim=6)
    model = FramePooledAudioModel(frame_pooler=pooler, embedding_dim=6)
    frame_features = torch.randn(15, 4, 7)

    speaker_embeddings = model(
        frame_features=frame_features,
        num_speakers=3,
        utterances_per_speaker=5,
    )

    assert speaker_embeddings.shape == (3, 6)


def test_frame_pooled_audio_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_frame_pooled_audio
model:
  family: frame_pooled_audio
  embedding_dim: 6
  frame_pooler:
    type: mean_std
    frame_dim: 4
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, FramePooledAudioModel)
    assert isinstance(model.frame_pooler, MeanStdFramePooler)
