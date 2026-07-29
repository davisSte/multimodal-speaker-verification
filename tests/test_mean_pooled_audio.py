from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import IdentityAudioEncoder
from multimodal_asv.models import MeanPooledAudioModel
from multimodal_asv.registry import build_model


def test_mean_pooled_audio_groups_flat_embeddings() -> None:
    model = MeanPooledAudioModel(embedding_dim=8)
    embeddings = torch.randn(15, 8)

    speaker_embeddings = model(
        audio_embeddings=embeddings,
        num_speakers=3,
        utterances_per_speaker=5,
    )

    assert speaker_embeddings.shape == (3, 8)
    assert torch.allclose(speaker_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_mean_pooled_audio_matches_manual_mean() -> None:
    model = MeanPooledAudioModel(embedding_dim=4, normalize=False)
    grouped = torch.tensor(
        [
            [[1.0, 3.0, 5.0, 7.0], [3.0, 5.0, 7.0, 9.0]],
            [[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]],
        ]
    )

    speaker_embeddings = model(audio_embeddings=grouped)

    expected = grouped.mean(dim=1)
    assert torch.allclose(speaker_embeddings, expected)


def test_mean_pooled_audio_builder_attaches_configured_audio_encoder(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_mean_pooled_audio
model:
  family: mean_pooled_audio
  audio_encoder:
    type: identity
  embedding_dim: 8
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, MeanPooledAudioModel)
    assert isinstance(model.audio_encoder, IdentityAudioEncoder)
