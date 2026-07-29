from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.encoders import IdentityAudioEncoder, StaticTextEncoder
from multimodal_asv.models import RJCAGlobalRepeatAudioLuarModel
from multimodal_asv.registry import build_model


def test_rjca_audio_global_luar_forward_from_embeddings() -> None:
    model = RJCAGlobalRepeatAudioLuarModel(
        audio_encoder=IdentityAudioEncoder(),
        text_encoder=StaticTextEncoder(embedding_dim=6),
        audio_dim=4,
        text_dim=6,
        final_dim=5,
        normalize=True,
    )
    audio_embeddings = torch.randn(3, 5, 4)
    text_embeddings = torch.randn(3, 5, 6)

    speaker_embeddings = model(
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
    )

    assert speaker_embeddings.shape == (3, 5)
    assert torch.allclose(speaker_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_rjca_audio_global_luar_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_rjca_audio_global_luar
model:
  family: rjca_audio_global_luar
  embedding_dim: 5
  audio_encoder:
    type: identity
  text_encoder:
    type: static
    embedding_dim: 6
  rjca:
    audio_dim: 4
    text_dim: 6
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, RJCAGlobalRepeatAudioLuarModel)
    assert isinstance(model.audio_encoder, IdentityAudioEncoder)
    assert isinstance(model.text_encoder, StaticTextEncoder)
