"""RJCA-style audio-text fusion models with global LUAR transcript context."""

from __future__ import annotations

from itertools import chain
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder, build_text_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class RJCALSTM(nn.Module):
    """Sequence encoder used inside the RJCA interaction stack."""

    def __init__(self, embed_size: int, dim: int, num_layers: int, dropout: float, residual_embeddings: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.rnn_dim = dim // 2
        self.linear = nn.Linear(dim + embed_size if residual_embeddings else dim, dim)
        self.rnn = nn.LSTM(
            embed_size,
            self.rnn_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.residual_embeddings = residual_embeddings
        self.init_hidden = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * 2 * num_layers, self.rnn_dim)))
        self.num_layers = num_layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.size(0)
        h0 = self.init_hidden[: 2 * self.num_layers].unsqueeze(1).expand(
            2 * self.num_layers, batch, self.rnn_dim
        ).contiguous()
        c0 = self.init_hidden[2 * self.num_layers :].unsqueeze(1).expand(
            2 * self.num_layers, batch, self.rnn_dim
        ).contiguous()
        outputs, _ = self.rnn(inputs, (h0, c0))
        if self.residual_embeddings:
            outputs = torch.cat([inputs, outputs], dim=-1)
        outputs = self.linear(self.dropout(outputs))
        return F.normalize(outputs, dim=-1)


class FeatureKeyDenseCoAttn(nn.Module):
    """Dense co-attention block over paired audio and text feature streams."""

    def __init__(self, dim1: int, dim2: int, dropout: float):
        super().__init__()
        dim = dim1 + dim2
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.query_linear = nn.Linear(dim, dim)
        self.key1_linear = nn.Linear(dim1, dim1)
        self.key2_linear = nn.Linear(dim2, dim2)
        self.value1_linear = nn.Linear(dim1, dim1)
        self.value2_linear = nn.Linear(dim2, dim2)
        self.relu = nn.ReLU()

    def forward(self, value1: torch.Tensor, value2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat((value1, value2), dim=-1)
        joint = self.query_linear(joint)
        key1 = self.key1_linear(value1).transpose(1, 2)
        key2 = self.key2_linear(value2).transpose(1, 2)
        value1 = self.value1_linear(value1)
        value2 = self.value2_linear(value2)
        weighted1, _ = self.qkv_attention(joint, key1, value1, dropout=self.dropouts[0])
        weighted2, _ = self.qkv_attention(joint, key2, value2, dropout=self.dropouts[1])
        return weighted1, weighted2

    def qkv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(key, query) / (query.size(-1) ** 0.5)
        scores = torch.tanh(scores)
        if dropout is not None:
            scores = dropout(scores)
        weighted = torch.tanh(torch.bmm(value, scores))
        return self.relu(weighted), scores


class FeatureKeyNormalSubLayer(nn.Module):
    def __init__(self, dim1: int, dim2: int, dropout: float):
        super().__init__()
        self.dense_coattn = FeatureKeyDenseCoAttn(dim1, dim2, dropout)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim1 + dim2, dim1), nn.ReLU(inplace=True), nn.Dropout(p=dropout)),
                nn.Sequential(nn.Linear(dim1 + dim2, dim2), nn.ReLU(inplace=True), nn.Dropout(p=dropout)),
            ]
        )

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted1, weighted2 = self.dense_coattn(data1, data2)
        data1 = data1 + self.linears[0](weighted1)
        data2 = data2 + self.linears[1](weighted2)
        return data1, data2


class FeatureKeyDCNLayer(nn.Module):
    def __init__(self, dim1: int, dim2: int, num_layers: int, dropout: float):
        super().__init__()
        self.dcn_layers = nn.ModuleList([FeatureKeyNormalSubLayer(dim1, dim2, dropout) for _ in range(num_layers)])

    def forward(self, data1: torch.Tensor, data2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2)
        return data1, data2


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        weights = self.attention(x)
        mu = torch.sum(x * weights, dim=2)
        sigma = torch.sqrt((torch.sum((x**2) * weights, dim=2) - mu**2).clamp(min=1e-4))
        return torch.cat((mu, sigma), dim=1)


class RJCAGlobalRepeatAudioLuarModel(nn.Module):
    """RJCA global-repeat model that repeats one grouped LUAR embedding across utterance positions."""

    def __init__(
        self,
        *,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        audio_dim: int = 192,
        text_dim: int = 512,
        final_dim: int = 192,
        rjca_layers: int = 2,
        rjca_dropout: float = 0.6,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        normalize: bool = True,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_dim = int(audio_dim)
        self.text_dim = int(text_dim)
        self.final_dim = int(final_dim)
        self.audio_extract = RJCALSTM(self.audio_dim, self.audio_dim, lstm_layers, lstm_dropout, residual_embeddings=True)
        self.text_global_proj = nn.Linear(self.text_dim, self.text_dim)
        self.text_extract = nn.Identity()
        self.coattn = FeatureKeyDCNLayer(self.text_dim, self.audio_dim, rjca_layers, rjca_dropout)
        self.joint_dim = self.audio_dim + self.text_dim
        self.joint = RJCALSTM(self.joint_dim, self.joint_dim, lstm_layers, dropout=0.0, residual_embeddings=True)
        self.asp = AttentiveStatisticsPooling(self.joint_dim)
        self.linear = nn.Linear(self.joint_dim * 2, self.final_dim)
        self.normalize = bool(normalize)
        self.speaker_loss = (
            AdditiveAngularMarginSoftmax(
                num_classes=num_classes,
                margin=speaker_loss_margin,
                scale=speaker_loss_scale,
                embedding_dim=self.final_dim,
            )
            if num_classes is not None
            else None
        )

    def forward(
        self,
        *,
        audio_embeddings: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        transcripts_grouped: Sequence[Sequence[str]] | None = None,
        padding_mask: torch.Tensor | None = None,
        num_speakers: int | None = None,
        utterances_per_speaker: int | None = None,
    ) -> torch.Tensor:
        if audio_embeddings is None:
            if waveforms is None:
                raise ValueError("Provide audio_embeddings or waveforms.")
            if waveforms.dim() != 3:
                raise ValueError("waveforms must have shape (speakers, utts, samples)")
            num_speakers = int(waveforms.size(0))
            utterances_per_speaker = int(waveforms.size(1))
            flat_waveforms = waveforms.reshape(num_speakers * utterances_per_speaker, waveforms.size(-1))
            flat_padding_mask = (
                padding_mask.reshape(num_speakers * utterances_per_speaker, padding_mask.size(-1))
                if padding_mask is not None
                else None
            )
            audio_embeddings = self.audio_encoder(flat_waveforms, padding_mask=flat_padding_mask)
            audio_embeddings = audio_embeddings.reshape(num_speakers, utterances_per_speaker, -1)

        if text_embeddings is None:
            if transcripts_grouped is None:
                raise ValueError("Provide text_embeddings or transcripts_grouped.")
            text_embeddings = self._encode_text_sequence(transcripts_grouped)

        return self.forward_from_embeddings(audio_embeddings, text_embeddings)

    def _encode_text_sequence(self, transcripts_grouped: Sequence[Sequence[str]]) -> torch.Tensor:
        lengths = [len(group) for group in transcripts_grouped]
        if not lengths:
            raise ValueError("rjca_audio_global_luar expects at least one speaker group.")
        if len(set(lengths)) != 1:
            raise ValueError(
                "rjca_audio_global_luar expects the same number of transcripts per speaker; "
                f"got lengths={lengths}."
            )
        text_embeddings = self.text_encoder(transcripts_grouped)
        if isinstance(text_embeddings, tuple):
            text_embeddings = text_embeddings[0]
        if text_embeddings.dim() != 2:
            raise ValueError(
                "rjca_audio_global_luar expects grouped LUAR embeddings shaped "
                f"(speakers, text_dim); got {tuple(text_embeddings.shape)}."
            )
        text_embeddings = self.text_global_proj(text_embeddings)
        return text_embeddings.unsqueeze(1).expand(-1, lengths[0], -1)

    def forward_from_embeddings(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if audio_embeddings.dim() != 3:
            raise ValueError("audio_embeddings must have shape (speakers, utts, audio_dim)")
        if text_embeddings.dim() != 3:
            raise ValueError("text_embeddings must have shape (speakers, utts, text_dim)")
        if audio_embeddings.shape[:2] != text_embeddings.shape[:2]:
            raise ValueError(
                "audio_embeddings and text_embeddings must agree on speakers/utts; "
                f"got {tuple(audio_embeddings.shape)} and {tuple(text_embeddings.shape)}."
            )
        target_dtype = self.linear.weight.dtype
        if audio_embeddings.dtype != target_dtype:
            audio_embeddings = audio_embeddings.to(target_dtype)
        if text_embeddings.dtype != target_dtype:
            text_embeddings = text_embeddings.to(target_dtype)

        audio = F.normalize(audio_embeddings, dim=-1)
        text = F.normalize(text_embeddings, dim=-1)
        audio = self.audio_extract(audio)
        text_ctx, audio_ctx = self.coattn(text, audio)
        joint_seq = torch.cat((text_ctx, audio_ctx), dim=-1)
        joint_seq = self.joint(joint_seq)
        pooled = self.asp(joint_seq)
        embeddings = self.linear(pooled)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


def build_rjca_audio_global_luar_model(config: ExperimentConfig) -> RJCAGlobalRepeatAudioLuarModel:
    """Build the released RJCA audio+global-LUAR model from config."""
    options = dict(config.model.options)
    speaker_loss = dict(options.pop("speaker_loss", {}) or {})
    rjca = dict(options.pop("rjca", {}) or {})
    audio_encoder = build_audio_encoder(
        with_default_download_root(
            options.pop("audio_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )
    text_encoder = build_text_encoder(
        with_default_download_root(
            options.pop("text_encoder", None),
            output_dir=config.output_dir,
            experiment_name=config.name,
        )
    )
    if audio_encoder is None:
        raise ValueError("rjca_audio_global_luar requires audio_encoder.")
    if text_encoder is None:
        raise ValueError("rjca_audio_global_luar requires text_encoder.")
    return RJCAGlobalRepeatAudioLuarModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        audio_dim=int(rjca.pop("audio_dim", 192)),
        text_dim=int(rjca.pop("text_dim", 512)),
        final_dim=int(options.pop("embedding_dim", 192)),
        rjca_layers=int(rjca.pop("layers", 2)),
        rjca_dropout=float(rjca.pop("dropout", 0.6)),
        lstm_layers=int(rjca.pop("lstm_layers", 2)),
        lstm_dropout=float(rjca.pop("lstm_dropout", 0.1)),
        normalize=bool(options.pop("normalize", True)),
        num_classes=_optional_int(options.pop("num_classes", None)),
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "RJCAGlobalRepeatAudioLuarModel",
    "build_rjca_audio_global_luar_model",
]
