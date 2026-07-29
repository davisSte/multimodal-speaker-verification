from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class MeanPooledAudioModel(nn.Module):
    """Audio-only baseline that averages utterance embeddings per speaker."""

    def __init__(
        self,
        *,
        audio_encoder: nn.Module | None = None,
        embedding_dim: int = 192,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
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
        audio_embeddings: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        num_speakers: int | None = None,
        utterances_per_speaker: int | None = None,
    ) -> torch.Tensor:
        if audio_embeddings is None:
            if self.audio_encoder is None:
                raise ValueError("Provide audio_embeddings or attach an audio_encoder.")
            if waveforms is None:
                raise ValueError("Provide waveforms when using audio_encoder.")
            if waveforms.dim() == 3:
                if num_speakers is None:
                    num_speakers = int(waveforms.size(0))
                if utterances_per_speaker is None:
                    utterances_per_speaker = int(waveforms.size(1))
                flat_waveforms = waveforms.reshape(num_speakers * utterances_per_speaker, waveforms.size(-1))
                flat_padding_mask = (
                    padding_mask.reshape(num_speakers * utterances_per_speaker, padding_mask.size(-1))
                    if padding_mask is not None
                    else None
                )
            else:
                flat_waveforms = waveforms
                flat_padding_mask = padding_mask
            audio_embeddings = self.audio_encoder(flat_waveforms, padding_mask=flat_padding_mask)

        grouped = self._group_embeddings(
            audio_embeddings,
            num_speakers=num_speakers,
            utterances_per_speaker=utterances_per_speaker,
        )
        speaker_embeddings = grouped.mean(dim=1)
        if self.normalize:
            speaker_embeddings = F.normalize(speaker_embeddings, dim=-1)
        return speaker_embeddings

    def _group_embeddings(
        self,
        audio_embeddings: torch.Tensor,
        *,
        num_speakers: int | None,
        utterances_per_speaker: int | None,
    ) -> torch.Tensor:
        if audio_embeddings.dim() == 3:
            return audio_embeddings
        if audio_embeddings.dim() != 2:
            raise ValueError("Expected audio_embeddings with shape (batch, dim) or (speakers, utterances, dim).")
        if audio_embeddings.size(-1) != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {audio_embeddings.size(-1)}.")
        if num_speakers is None or utterances_per_speaker is None:
            raise ValueError("num_speakers and utterances_per_speaker are required for flat embeddings.")
        expected = num_speakers * utterances_per_speaker
        if audio_embeddings.size(0) != expected:
            raise ValueError(f"Expected {expected} flat embeddings, got {audio_embeddings.size(0)}.")
        return audio_embeddings.reshape(num_speakers, utterances_per_speaker, self.embedding_dim)


def build_mean_pooled_audio_model(config: ExperimentConfig) -> MeanPooledAudioModel:
    model_options = dict(config.model.options)
    speaker_loss = dict(model_options.pop("speaker_loss", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            model_options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )

    return MeanPooledAudioModel(
        audio_encoder=audio_encoder,
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
    "MeanPooledAudioModel",
    "build_mean_pooled_audio_model",
]
