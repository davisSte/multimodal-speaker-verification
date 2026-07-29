"""Audio-only multi-utterance models with learned query-attention aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class QueryAttentionAggregation(nn.Module):
    """Pool utterance embeddings with a single learned query."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        temperature: float = 0.3,
        query_init_scale: float = 0.02,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        nn.init.normal_(self.query, std=query_init_scale)

    def forward(
        self,
        utterance_embeddings: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if utterance_embeddings.dim() != 3:
            raise ValueError("Expected utterance_embeddings with shape (speakers, utterances, dim).")

        num_speakers, _num_utterances, embedding_dim = utterance_embeddings.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {embedding_dim}.")

        keys = utterance_embeddings.transpose(0, 1).contiguous()
        query = self.query.expand(-1, num_speakers, -1) / self.temperature
        attended, weights = self.attention(query, keys, keys, need_weights=return_weights)
        speaker_embeddings = attended.squeeze(0)
        if self.normalize:
            speaker_embeddings = F.normalize(speaker_embeddings, dim=-1)

        if not return_weights:
            return speaker_embeddings
        if weights is None:
            raise RuntimeError("Attention weights were requested but not returned.")
        return speaker_embeddings, weights.squeeze(1)


class QueryAttentionAudioModel(nn.Module):
    """
    Audio-only speaker model for query attention over utterance embeddings.

    The audio encoder is optional so this model can be tested with precomputed
    utterance embeddings before the WavLM/ECAPA encoder is migrated.
    """

    def __init__(
        self,
        *,
        audio_encoder: nn.Module | None = None,
        embedding_dim: int = 192,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.3,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.embedding_dim = int(embedding_dim)
        self.aggregation = QueryAttentionAggregation(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            temperature=temperature,
            normalize=normalize,
        )
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
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        return self.aggregation(grouped, return_weights=return_weights)

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


def build_query_attention_audio_model(config: ExperimentConfig) -> QueryAttentionAudioModel:
    """Build the paper-facing query-attention audio model from config."""
    model_options = dict(config.model.options)
    attention = dict(model_options.pop("attention", {}) or {})
    speaker_loss = dict(model_options.pop("speaker_loss", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            model_options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )

    return QueryAttentionAudioModel(
        audio_encoder=audio_encoder,
        embedding_dim=int(model_options.pop("embedding_dim", 192)),
        num_heads=int(attention.pop("heads", 4)),
        dropout=float(attention.pop("dropout", 0.1)),
        temperature=float(attention.pop("temperature", 0.3)),
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
    "QueryAttentionAggregation",
    "QueryAttentionAudioModel",
    "build_query_attention_audio_model",
]
