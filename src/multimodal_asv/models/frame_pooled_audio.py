"""Audio-only models that pool frame-level acoustic evidence across utterances."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class MeanStdFramePooler(nn.Module):
    """Small fallback frame pooler used for tests and precomputed frame maps."""

    def __init__(self, *, frame_dim: int, embedding_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.embedding_dim = int(embedding_dim)
        self.normalize = bool(normalize)
        self.projection = nn.Linear(self.frame_dim * 2, self.embedding_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.dim() != 3:
            raise ValueError("Expected frame_features with shape (speakers, frame_dim, frames).")
        if frame_features.size(1) != self.frame_dim:
            raise ValueError(f"Expected frame dim {self.frame_dim}, got {frame_features.size(1)}.")
        mean = frame_features.mean(dim=-1)
        std = frame_features.std(dim=-1, unbiased=False).clamp(min=1e-4)
        embeddings = self.projection(torch.cat([mean, std], dim=-1))
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class FramePooledAudioModel(nn.Module):
    """
    Audio-only frame-level aggregation.

    The model concatenates ECAPA frame maps from multiple utterances for each
    speaker, then pools once over the combined timeline.
    """

    def __init__(
        self,
        *,
        audio_encoder: nn.Module | None = None,
        frame_pooler: nn.Module | None = None,
        embedding_dim: int = 192,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.frame_pooler = frame_pooler
        self.embedding_dim = int(embedding_dim)
        self.normalize = bool(normalize)
        self.speaker_loss = (
            AdditiveAngularMarginSoftmax(
                num_classes=num_classes,
                margin=speaker_loss_margin,
                scale=speaker_loss_scale,
                embedding_dim=embedding_dim,
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
        num_speakers: int | None = None,
        utterances_per_speaker: int | None = None,
    ) -> torch.Tensor:
        if frame_features is None:
            frame_features = self._encode_frame_features(
                waveforms=waveforms,
                padding_mask=padding_mask,
                num_speakers=num_speakers,
                utterances_per_speaker=utterances_per_speaker,
            )

        concat_frames = self._concat_speaker_frames(
            frame_features,
            num_speakers=num_speakers,
            utterances_per_speaker=utterances_per_speaker,
        )
        embeddings = self._pool_frames(concat_frames)
        if embeddings.size(-1) != self.embedding_dim:
            raise ValueError(f"Expected pooled embedding dim {self.embedding_dim}, got {embeddings.size(-1)}.")
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def _encode_frame_features(
        self,
        *,
        waveforms: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        num_speakers: int | None,
        utterances_per_speaker: int | None,
    ) -> torch.Tensor:
        if self.audio_encoder is None:
            raise ValueError("Provide frame_features or attach an audio_encoder.")
        if waveforms is None:
            raise ValueError("Provide waveforms when using audio_encoder.")

        if waveforms.dim() == 3:
            num_speakers = int(waveforms.size(0))
            utterances_per_speaker = int(waveforms.size(1))
            flat_waveforms = waveforms.reshape(num_speakers * utterances_per_speaker, waveforms.size(-1))
        elif waveforms.dim() == 2:
            if num_speakers is None or utterances_per_speaker is None:
                raise ValueError("num_speakers and utterances_per_speaker are required for flat waveforms.")
            flat_waveforms = waveforms
        else:
            raise ValueError("Expected waveforms with shape (speakers, utterances, samples) or (batch, samples).")

        return self.audio_encoder(
            flat_waveforms,
            padding_mask=padding_mask,
            return_frame_features=True,
        )

    @staticmethod
    def _concat_speaker_frames(
        frame_features: torch.Tensor,
        *,
        num_speakers: int | None,
        utterances_per_speaker: int | None,
    ) -> torch.Tensor:
        if frame_features.dim() == 4:
            speakers, utterances, channels, frames = frame_features.shape
            return frame_features.permute(0, 2, 1, 3).contiguous().view(speakers, channels, utterances * frames)
        if frame_features.dim() != 3:
            raise ValueError("Expected frame_features with shape (speakers, utterances, channels, frames) or (batch, channels, frames).")
        if num_speakers is None or utterances_per_speaker is None:
            raise ValueError("num_speakers and utterances_per_speaker are required for flat frame_features.")
        expected = int(num_speakers) * int(utterances_per_speaker)
        if frame_features.size(0) != expected:
            raise ValueError(f"Expected {expected} flat frame maps, got {frame_features.size(0)}.")
        channels = frame_features.size(1)
        frames = frame_features.size(2)
        grouped = frame_features.view(int(num_speakers), int(utterances_per_speaker), channels, frames)
        return grouped.permute(0, 2, 1, 3).contiguous().view(int(num_speakers), channels, int(utterances_per_speaker) * frames)

    def _pool_frames(self, concat_frames: torch.Tensor) -> torch.Tensor:
        if self.frame_pooler is not None:
            return self.frame_pooler(concat_frames)
        backend = getattr(self.audio_encoder, "backend", None)
        if backend is not None and hasattr(backend, "pool_frame_features"):
            return backend.pool_frame_features(concat_frames)
        raise ValueError("Attach frame_pooler or use an audio_encoder with backend.pool_frame_features.")


def build_frame_pooled_audio_model(config: ExperimentConfig) -> FramePooledAudioModel:
    """Build the frame-concat audio model used by the released configs."""
    model_options = dict(config.model.options)
    speaker_loss = dict(model_options.pop("speaker_loss", {}) or {})
    pooler_options = dict(model_options.pop("frame_pooler", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            model_options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )
    frame_pooler = None
    if pooler_options:
        pooler_type = str(pooler_options.pop("type", "mean_std"))
        if pooler_type != "mean_std":
            raise ValueError(f"Unknown frame_pooler type: {pooler_type!r}")
        frame_pooler = MeanStdFramePooler(
            frame_dim=int(pooler_options.pop("frame_dim")),
            embedding_dim=int(model_options.get("embedding_dim", 192)),
            normalize=bool(pooler_options.pop("normalize", True)),
        )

    return FramePooledAudioModel(
        audio_encoder=audio_encoder,
        frame_pooler=frame_pooler,
        embedding_dim=int(model_options.pop("embedding_dim", 192)),
        normalize=bool(model_options.pop("normalize", True)),
        num_classes=_optional_int(model_options.pop("num_classes", None)),
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "FramePooledAudioModel",
    "MeanStdFramePooler",
    "build_frame_pooled_audio_model",
]
