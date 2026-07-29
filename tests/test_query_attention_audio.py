from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import IdentityAudioEncoder
from multimodal_asv.models import QueryAttentionAggregation, QueryAttentionAudioModel
from multimodal_asv.registry import build_model


def test_query_attention_aggregation_shapes() -> None:
    model = QueryAttentionAggregation(embedding_dim=8, num_heads=2, dropout=0.0)
    embeddings = torch.randn(3, 5, 8)

    speaker_embeddings, weights = model(embeddings, return_weights=True)

    assert speaker_embeddings.shape == (3, 8)
    assert weights.shape == (3, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_query_attention_audio_model_groups_flat_embeddings() -> None:
    model = QueryAttentionAudioModel(embedding_dim=8, num_heads=2, dropout=0.0)
    embeddings = torch.randn(15, 8)

    speaker_embeddings = model(
        audio_embeddings=embeddings,
        num_speakers=3,
        utterances_per_speaker=5,
    )

    assert speaker_embeddings.shape == (3, 8)


def test_query_attention_builder_attaches_configured_audio_encoder(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_query_attention_audio
model:
  family: query_attention_audio
  audio_encoder:
    type: identity
  embedding_dim: 8
  attention:
    heads: 2
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, QueryAttentionAudioModel)
    assert isinstance(model.audio_encoder, IdentityAudioEncoder)
