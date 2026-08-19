from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import IdentityAudioEncoder
from multimodal_asv.models import UtteranceAudioProsodyModel
from multimodal_asv.registry import build_model


def test_utterance_audio_prosody_shape_and_checkpoint_keys() -> None:
    model = UtteranceAudioProsodyModel(
        audio_encoder=IdentityAudioEncoder(),
        audio_dim=4,
        prosody_dim=3,
        prosody_projection_dim=2,
        final_dim=4,
        pre_concat_layernorm=True,
    )

    fused = model(
        waveforms=torch.randn(3, 4),
        pitch_features=torch.randn(3, 3),
    )

    assert fused.shape == (3, 4)
    assert torch.allclose(fused.norm(dim=-1), torch.ones(3), atol=1e-6)
    keys = set(model.state_dict())
    assert "pitch_proj.weight" in keys
    assert "pitch_pre_concat_ln.weight" in keys
    assert "fusion_proj.weight" in keys
    assert "fusion.projections.pitch.weight" not in keys


def test_utterance_audio_prosody_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_utterance_audio_prosody
model:
  family: utterance_audio_prosody
  audio_encoder:
    type: identity
  fusion:
    modalities: [audio, pitch]
    audio_dim: 4
    pitch_dim: 3
    pitch_projection_dim: 2
    final_dim: 4
    pre_concat_layernorm: true
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, UtteranceAudioProsodyModel)
    assert isinstance(model.audio_encoder, IdentityAudioEncoder)
    assert model.embedding_dim == 4
