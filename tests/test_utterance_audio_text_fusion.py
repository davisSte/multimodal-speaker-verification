from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import IdentityAudioEncoder, StaticTextEncoder
from multimodal_asv.models import ProjectedConcatFusion, UtteranceAudioTextFusionModel
from multimodal_asv.registry import build_model


def test_projected_concat_audio_text_fusion_shape() -> None:
    fusion = ProjectedConcatFusion(
        input_dims={"audio": 4, "text": 6},
        projection_dim=8,
        final_dim=5,
        pre_concat_layernorm=True,
    )
    model = UtteranceAudioTextFusionModel(fusion=fusion)

    fused = model(
        audio_embeddings=torch.randn(3, 4),
        text_embeddings=torch.randn(3, 6),
    )

    assert fused.shape == (3, 5)
    assert torch.allclose(fused.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_projected_concat_audio_pitch_fusion_shape() -> None:
    fusion = ProjectedConcatFusion(
        input_dims={"audio": 4, "pitch": 2},
        projection_dim=8,
        final_dim=5,
    )
    model = UtteranceAudioTextFusionModel(fusion=fusion)

    fused = model(
        audio_embeddings=torch.randn(3, 4),
        pitch_features=torch.randn(3, 2),
    )

    assert fused.shape == (3, 5)


def test_utterance_fusion_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_utterance_audio_text
model:
  family: utterance_audio_text_fusion
  audio_encoder:
    type: identity
  text_encoder:
    type: static
    embedding_dim: 6
  fusion:
    modalities: [audio, text]
    audio_dim: 4
    text_dim: 6
    projection_dim: 8
    final_dim: 5
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, UtteranceAudioTextFusionModel)
    assert isinstance(model.audio_encoder, IdentityAudioEncoder)
    assert isinstance(model.text_encoder, StaticTextEncoder)
    assert model.embedding_dim == 5


def test_utterance_fusion_uses_encoder_interfaces() -> None:
    fusion = ProjectedConcatFusion(
        input_dims={"audio": 4, "text": 6},
        projection_dim=8,
        final_dim=5,
    )
    model = UtteranceAudioTextFusionModel(
        audio_encoder=IdentityAudioEncoder(),
        text_encoder=StaticTextEncoder(embedding_dim=6),
        fusion=fusion,
    )

    fused = model(
        waveforms=torch.randn(2, 4),
        transcripts=["hello", "speaker"],
    )

    assert fused.shape == (2, 5)


def test_utterance_fusion_builder_attaches_speaker_loss(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_utterance_audio_text_loss
model:
  family: utterance_audio_text_fusion
  num_classes: 3
  audio_encoder:
    type: identity
  text_encoder:
    type: static
    embedding_dim: 6
  fusion:
    modalities: [audio, text]
    audio_dim: 4
    text_dim: 6
    projection_dim: 8
    final_dim: 5
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert model.speaker_loss is not None
    assert model.embedding_dim == 5
