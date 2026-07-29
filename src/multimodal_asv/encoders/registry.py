from __future__ import annotations

from typing import Any

import torch.nn as nn

from .identity import IdentityAudioEncoder
from .identity import StaticTextEncoder


def build_audio_encoder(config: Any) -> nn.Module | None:
    """Build an audio encoder from a config section."""

    if config is None:
        return None
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, dict):
        raise ValueError("audio_encoder must be a string, mapping, or null.")

    encoder_type = str(config.get("type", "")).strip()
    if encoder_type == "identity":
        return IdentityAudioEncoder()
    if encoder_type == "wavlm_ecapa":
        # Lazy import keeps package-level encoder imports lightweight.
        from .audio import build_wavlm_ecapa_encoder

        return build_wavlm_ecapa_encoder({key: value for key, value in config.items() if key != "type"})
    raise ValueError(f"Unknown audio encoder type: {encoder_type!r}")


def build_text_encoder(config: Any) -> nn.Module | None:
    """Build a text encoder from a config section."""

    if config is None:
        return None
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, dict):
        raise ValueError("text_encoder must be a string, mapping, or null.")

    encoder_type = str(config.get("type", "")).strip()
    if encoder_type == "static":
        return StaticTextEncoder(embedding_dim=int(config.get("embedding_dim", 512)))
    if encoder_type == "luar":
        from .text import build_luar_encoder

        return build_luar_encoder({key: value for key, value in config.items() if key != "type"})
    raise ValueError(f"Unknown text encoder type: {encoder_type!r}")


__all__ = ["build_audio_encoder", "build_text_encoder"]
