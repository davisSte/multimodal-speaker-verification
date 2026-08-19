"""Utterance-level audio+prosody fusion for pitch-rate-voicing models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class UtteranceAudioProsodyModel(nn.Module):
    """Audio plus pitch/rate/voicing fusion."""

    def __init__(
        self,
        *,
        audio_encoder: nn.Module | None = None,
        audio_dim: int = 192,
        prosody_dim: int = 3,
        prosody_projection_dim: int = 32,
        final_dim: int = 192,
        pre_concat_layernorm: bool = False,
        normalize_pre_concat: bool = False,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.embedding_dim = int(final_dim)
        self.audio_proj = (
            nn.Identity()
            if int(audio_dim) == self.embedding_dim
            else nn.Linear(int(audio_dim), self.embedding_dim)
        )
        self.pitch_proj = nn.Linear(int(prosody_dim), int(prosody_projection_dim))
        self.audio_pre_concat_ln = nn.LayerNorm(self.embedding_dim) if pre_concat_layernorm else nn.Identity()
        self.pitch_pre_concat_ln = nn.LayerNorm(int(prosody_projection_dim)) if pre_concat_layernorm else nn.Identity()
        self.fusion_proj = nn.Linear(self.embedding_dim + int(prosody_projection_dim), self.embedding_dim)
        self.normalize_pre_concat = bool(normalize_pre_concat)
        self.normalize = bool(normalize)
        self.last_fusion_stats: dict[str, float] = {}
        self.speaker_loss = (
            AdditiveAngularMarginSoftmax(
                num_classes=num_classes,
                margin=speaker_loss_margin,
                scale=speaker_loss_scale,
                embedding_dim=self.embedding_dim,
            )
            if num_classes is not None
            else None
        )

    def forward(
        self,
        *,
        audio_embeddings: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        pitch_features: torch.Tensor | None = None,
        transcripts: list[str] | None = None,
    ) -> torch.Tensor:
        del transcripts
        if audio_embeddings is None:
            if self.audio_encoder is None or waveforms is None:
                raise ValueError("Provide audio_embeddings or attach audio_encoder and waveforms.")
            audio_embeddings = self.audio_encoder(waveforms, padding_mask=padding_mask)
        if pitch_features is None:
            raise ValueError("Provide pitch_features for audio+prosody fusion.")

        pitch_features = pitch_features.to(audio_embeddings.device, dtype=audio_embeddings.dtype)
        target_dtype = self.fusion_proj.weight.dtype
        if audio_embeddings.dtype != target_dtype:
            audio_embeddings = audio_embeddings.to(target_dtype)
        if pitch_features.dtype != target_dtype:
            pitch_features = pitch_features.to(target_dtype)

        audio_proj = self.audio_pre_concat_ln(self.audio_proj(audio_embeddings))
        pitch_proj = self.pitch_pre_concat_ln(self.pitch_proj(pitch_features))
        if self.normalize_pre_concat:
            audio_proj = F.normalize(audio_proj, dim=-1)
            pitch_proj = F.normalize(pitch_proj, dim=-1)
        joint = torch.cat([audio_proj, pitch_proj], dim=-1)
        fused = self.fusion_proj(joint)
        with torch.no_grad():
            self.last_fusion_stats = {
                "audio_norm": float(audio_proj.norm(dim=-1).mean().item()),
                "pitch_norm": float(pitch_proj.norm(dim=-1).mean().item()),
                "joint_norm": float(joint.norm(dim=-1).mean().item()),
            }
        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused


def build_utterance_audio_prosody_model(config: ExperimentConfig) -> UtteranceAudioProsodyModel:
    """Build an utterance-level audio+prosody model."""
    options = dict(config.model.options)
    prosody = dict(options.pop("prosody", {}) or {})
    fusion = dict(options.pop("fusion", {}) or {})
    speaker_loss = dict(options.pop("speaker_loss", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )
    num_classes = options.pop("num_classes", None)
    return UtteranceAudioProsodyModel(
        audio_encoder=audio_encoder,
        audio_dim=int(fusion.pop("audio_dim", options.pop("embedding_dim", 192))),
        prosody_dim=int(prosody.pop("feature_dim", fusion.pop("pitch_dim", 3))),
        prosody_projection_dim=int(prosody.pop("projection_dim", fusion.pop("pitch_projection_dim", 32))),
        final_dim=int(fusion.pop("final_dim", 192)),
        pre_concat_layernorm=bool(fusion.pop("pre_concat_layernorm", prosody.pop("pre_concat_layernorm", False))),
        normalize_pre_concat=bool(fusion.pop("normalize_pre_concat", False)),
        normalize=bool(fusion.pop("normalize", options.pop("normalize", True))),
        num_classes=int(num_classes) if num_classes is not None else None,
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )
__all__ = ["UtteranceAudioProsodyModel", "build_utterance_audio_prosody_model"]
