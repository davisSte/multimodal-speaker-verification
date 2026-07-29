"""Frame-level audio-text fusion models for grouped multi-utterance episodes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import TextEncoder, build_audio_encoder, build_text_encoder
from multimodal_asv.models.frame_pooled_audio import FramePooledAudioModel, MeanStdFramePooler
from multimodal_asv.models.utterance_audio_text_fusion import ProjectedConcatFusion
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class WeightedSumFusion(nn.Module):
    """Project audio/text branches to a shared space and combine fixed weights."""

    def __init__(
        self,
        *,
        audio_dim: int,
        text_dim: int,
        final_dim: int = 192,
        audio_weight: float = 0.5,
        text_weight: float = 0.5,
        pre_fusion_layernorm: bool = False,
        normalize_branches: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, final_dim)
        self.text_proj = nn.Linear(text_dim, final_dim)
        self.audio_layer_norm = nn.LayerNorm(final_dim) if pre_fusion_layernorm else nn.Identity()
        self.text_layer_norm = nn.LayerNorm(final_dim) if pre_fusion_layernorm else nn.Identity()
        self.audio_weight = float(audio_weight)
        self.text_weight = float(text_weight)
        self.normalize_branches = bool(normalize_branches)
        self.normalize = bool(normalize)
        self.last_fusion_stats: dict[str, float] = {}

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        audio = self.audio_layer_norm(self.audio_proj(features["audio"]))
        text = self.text_layer_norm(self.text_proj(features["text"]))
        if self.normalize_branches:
            audio = F.normalize(audio, dim=-1)
            text = F.normalize(text, dim=-1)
        fused = (self.audio_weight * audio) + (self.text_weight * text)
        with torch.no_grad():
            self.last_fusion_stats = {
                "audio_norm": float(audio.norm(dim=-1).mean().item()),
                "text_norm": float(text.norm(dim=-1).mean().item()),
                "audio_weight": self.audio_weight,
                "text_weight": self.text_weight,
            }
        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused


class FramePooledAudioTextModel(nn.Module):
    """
    Frame-level audio aggregation fused with grouped text embeddings.

    Audio is pooled after concatenating frame maps across a speaker's utterances.
    Text is encoded as a grouped speaker episode, matching LUAR-style usage.
    """

    def __init__(
        self,
        *,
        audio_model: FramePooledAudioModel,
        text_encoder: TextEncoder,
        fusion: nn.Module,
        embedding_dim: int | None = None,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_model = audio_model
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.embedding_dim = int(embedding_dim if embedding_dim is not None else _infer_fusion_output_dim(fusion))
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

    @property
    def audio_encoder(self) -> nn.Module | None:
        return self.audio_model.audio_encoder

    def forward(
        self,
        *,
        frame_features: torch.Tensor | None = None,
        audio_embeddings: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        transcripts_grouped: Sequence[Sequence[str]] | None = None,
        padding_mask: torch.Tensor | None = None,
        num_speakers: int | None = None,
        utterances_per_speaker: int | None = None,
    ) -> torch.Tensor:
        if audio_embeddings is None:
            audio_embeddings = self.audio_model(
                frame_features=frame_features,
                waveforms=waveforms,
                padding_mask=padding_mask,
                num_speakers=num_speakers,
                utterances_per_speaker=utterances_per_speaker,
            )
        if text_embeddings is None:
            if transcripts_grouped is None:
                raise ValueError("Provide text_embeddings or transcripts_grouped.")
            encoded = self.text_encoder(transcripts_grouped)
            text_embeddings = encoded[0] if isinstance(encoded, tuple) else encoded
        if audio_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError(
                f"Audio/text speaker counts must match, got {audio_embeddings.size(0)} and {text_embeddings.size(0)}."
            )
        text_embeddings = text_embeddings.to(device=audio_embeddings.device, dtype=audio_embeddings.dtype)
        return self.fusion({"audio": audio_embeddings, "text": text_embeddings})


def build_frame_pooled_audio_text_model(config: ExperimentConfig) -> FramePooledAudioTextModel:
    """Build frame-level audio-text fusion from a paper-facing config."""
    options = dict(config.model.options)
    speaker_loss = dict(options.pop("speaker_loss", {}) or {})
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
    if text_encoder is None:
        raise ValueError("frame_pooled_audio_text requires text_encoder.")

    audio_options = dict(options.pop("audio", {}) or {})
    pooler_options = dict(options.pop("frame_pooler", {}) or {})
    embedding_dim = int(audio_options.pop("embedding_dim", options.get("audio_dim", 192)))
    frame_pooler = _build_frame_pooler(pooler_options, embedding_dim=embedding_dim)
    audio_model = FramePooledAudioModel(
        audio_encoder=audio_encoder,
        frame_pooler=frame_pooler,
        embedding_dim=embedding_dim,
        normalize=bool(audio_options.pop("normalize", True)),
    )
    fusion = _build_fusion(dict(options.pop("fusion", {}) or {}))
    final_dim = _infer_fusion_output_dim(fusion)
    return FramePooledAudioTextModel(
        audio_model=audio_model,
        text_encoder=text_encoder,
        fusion=fusion,
        embedding_dim=final_dim,
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


def _build_fusion(options: dict[str, object]) -> nn.Module:
    fusion_type = str(options.pop("type", "projected_concat"))
    audio_dim = int(options.pop("audio_dim", 192))
    text_dim = int(options.pop("text_dim", 512))
    final_dim = int(options.pop("final_dim", 192))
    normalize = bool(options.pop("normalize", True))
    if fusion_type == "projected_concat":
        return ProjectedConcatFusion(
            input_dims={"audio": audio_dim, "text": text_dim},
            projection_dim=int(options.pop("projection_dim", 256)),
            final_dim=final_dim,
            pre_concat_layernorm=bool(options.pop("pre_concat_layernorm", False)),
            normalize_pre_concat=bool(options.pop("normalize_pre_concat", False)),
            normalize=normalize,
        )
    if fusion_type == "weighted_sum":
        return WeightedSumFusion(
            audio_dim=audio_dim,
            text_dim=text_dim,
            final_dim=final_dim,
            audio_weight=float(options.pop("audio_weight", 0.5)),
            text_weight=float(options.pop("text_weight", 0.5)),
            pre_fusion_layernorm=bool(options.pop("pre_fusion_layernorm", False)),
            normalize_branches=bool(options.pop("normalize_branches", False)),
            normalize=normalize,
        )
    raise ValueError(f"Unknown frame_pooled_audio_text fusion type: {fusion_type!r}")


def _infer_fusion_output_dim(fusion: nn.Module) -> int:
    if isinstance(fusion, ProjectedConcatFusion):
        return int(fusion.output.out_features)
    if isinstance(fusion, WeightedSumFusion):
        return int(fusion.audio_proj.out_features)
    raise ValueError(f"Unsupported fusion module for output-dim inference: {type(fusion).__name__}")


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "FramePooledAudioTextModel",
    "WeightedSumFusion",
    "build_frame_pooled_audio_text_model",
]
