from __future__ import annotations

from multimodal_asv.encoders import StaticTextEncoder, build_audio_encoder, build_text_encoder
from multimodal_asv.encoders import IdentityAudioEncoder


def test_build_identity_audio_encoder() -> None:
    encoder = build_audio_encoder({"type": "identity"})

    assert isinstance(encoder, IdentityAudioEncoder)


def test_build_static_text_encoder() -> None:
    encoder = build_text_encoder({"type": "static", "embedding_dim": 7})

    assert isinstance(encoder, StaticTextEncoder)
    assert encoder(["hello"]).shape == (1, 7)
