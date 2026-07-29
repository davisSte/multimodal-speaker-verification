"""Frame-level audio-prosody fusion models built on pooled acoustic frame maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder
from multimodal_asv.models.frame_pooled_audio import FramePooledAudioModel, MeanStdFramePooler
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class FramePooledAudioProsodyModel(nn.Module):
    """Frame-level audio aggregation fused with grouped prosody statistics."""

    def __init__(
        self,
        *,
        audio_model: FramePooledAudioModel,
        prosody_dim: int = 3,
        prosody_projection_dim: int = 32,
        embedding_dim: int = 192,
        pre_concat_layernorm: bool = True,
        normalize_pre_concat: bool = False,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.embedding_dim = int(embedding_dim)
        self.prosody_dim = int(prosody_dim)
        self.prosody_projection_dim = int(prosody_projection_dim)
        self.audio_proj = nn.Identity()
        self.prosody_proj = nn.Linear(self.prosody_dim * 2, self.prosody_projection_dim)
        self.audio_pre_concat_ln = nn.LayerNorm(self.embedding_dim) if pre_concat_layernorm else nn.Identity()
        self.prosody_pre_concat_ln = nn.LayerNorm(self.prosody_projection_dim) if pre_concat_layernorm else nn.Identity()
        self.fusion_proj = nn.Linear(self.embedding_dim + self.prosody_projection_dim, self.embedding_dim)
        self.normalize_pre_concat = bool(normalize_pre_concat)
        self.normalize = bool(normalize)
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
        frame_features: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        pitch_features_grouped: torch.Tensor | None = None,
        num_speakers: int | None = None,
        utterances_per_speaker: int | None = None,
    ) -> torch.Tensor:
        if pitch_features_grouped is None:
            raise ValueError("pitch_features_grouped is required for frame-level audio+prosody fusion.")

        audio_emb = self.audio_model(
            frame_features=frame_features,
            waveforms=waveforms,
            padding_mask=padding_mask,
            num_speakers=num_speakers,
            utterances_per_speaker=utterances_per_speaker,
        )
        prosody_stats = self._encode_grouped_prosody_stats(
            pitch_features_grouped.to(audio_emb.device, dtype=audio_emb.dtype)
        )

        target_dtype = self.fusion_proj.weight.dtype
        if audio_emb.dtype != target_dtype:
            audio_emb = audio_emb.to(target_dtype)
        if prosody_stats.dtype != target_dtype:
            prosody_stats = prosody_stats.to(target_dtype)

        audio_proj = self.audio_pre_concat_ln(self.audio_proj(audio_emb))
        prosody_proj = self.prosody_pre_concat_ln(self.prosody_proj(prosody_stats))
        if self.normalize_pre_concat:
            audio_proj = F.normalize(audio_proj, dim=-1)
            prosody_proj = F.normalize(prosody_proj, dim=-1)
        fused = self.fusion_proj(torch.cat([audio_proj, prosody_proj], dim=-1))
        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused

    @staticmethod
    def _encode_grouped_prosody_stats(pitch_features_grouped: torch.Tensor) -> torch.Tensor:
        if pitch_features_grouped.dim() != 3:
            raise ValueError(
                "pitch_features_grouped must have shape (speakers, utts, feature_dim), "
                f"got {tuple(pitch_features_grouped.shape)}"
            )
        mean = pitch_features_grouped.mean(dim=1)
        std = pitch_features_grouped.std(dim=1, unbiased=False)
        return torch.cat([mean, std], dim=-1)


def build_frame_pooled_audio_prosody_model(config: ExperimentConfig) -> FramePooledAudioProsodyModel:
    """Build frame-level audio+prosody fusion from a paper-facing config."""
    options = dict(config.model.options)
    speaker_loss = dict(options.pop("speaker_loss", {}) or {})
    prosody = dict(options.pop("prosody", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )
    embedding_dim = int(options.pop("embedding_dim", 192))
    frame_pooler = _build_frame_pooler(dict(options.pop("frame_pooler", {}) or {}), embedding_dim=embedding_dim)
    audio_model = FramePooledAudioModel(
        audio_encoder=audio_encoder,
        frame_pooler=frame_pooler,
        embedding_dim=embedding_dim,
        normalize=bool(options.pop("normalize", True)),
    )
    return FramePooledAudioProsodyModel(
        audio_model=audio_model,
        prosody_dim=int(prosody.pop("feature_dim", 3)),
        prosody_projection_dim=int(prosody.pop("projection_dim", 32)),
        embedding_dim=embedding_dim,
        pre_concat_layernorm=bool(prosody.pop("pre_concat_layernorm", True)),
        normalize_pre_concat=bool(prosody.pop("normalize_pre_concat", False)),
        normalize=bool(options.pop("normalize", True)),
        num_classes=_optional_int(options.pop("num_classes", None)),
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )


def _build_frame_pooler(options: dict[str, object], *, embedding_dim: int) -> nn.Module | None:
    if not options:
        return None
    pooler_type = str(options.pop("type", "mean_std"))
    if pooler_type != "mean_std":
        raise ValueError(f"Unknown frame_pooler type: {pooler_type!r}")
    return MeanStdFramePooler(
        frame_dim=int(options.pop("frame_dim")),
        embedding_dim=embedding_dim,
        normalize=bool(options.pop("normalize", True)),
    )


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "FramePooledAudioProsodyModel",
    "build_frame_pooled_audio_prosody_model",
]
