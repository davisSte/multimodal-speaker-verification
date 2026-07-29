"""Utterance-level multimodal fusion for audio+text and audio+prosody models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import AudioEncoder, TextEncoder, build_audio_encoder, build_text_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class ProjectedConcatFusion(nn.Module):
    """Project modality embeddings, concatenate them, and map to a speaker embedding."""

    def __init__(
        self,
        *,
        input_dims: dict[str, int],
        projection_dim: int = 256,
        final_dim: int = 192,
        pre_concat_layernorm: bool = False,
        normalize_pre_concat: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("ProjectedConcatFusion requires at least two modalities.")
        self.input_dims = dict(input_dims)
        self.normalize_pre_concat = bool(normalize_pre_concat)
        self.normalize = bool(normalize)
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, projection_dim) for name, dim in self.input_dims.items()}
        )
        self.layer_norms = nn.ModuleDict(
            {
                name: nn.LayerNorm(projection_dim) if pre_concat_layernorm else nn.Identity()
                for name in self.input_dims
            }
        )
        self.output = nn.Linear(projection_dim * len(self.input_dims), final_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for name in self.input_dims:
            if name not in features:
                raise ValueError(f"Missing modality feature {name!r}.")
            value = features[name]
            expected_dim = self.input_dims[name]
            if value.size(-1) != expected_dim:
                raise ValueError(f"Expected {name!r} dim {expected_dim}, got {value.size(-1)}.")
            value = self.layer_norms[name](self.projections[name](value))
            if self.normalize_pre_concat:
                value = F.normalize(value, dim=-1)
            projected.append(value)
        fused = self.output(torch.cat(projected, dim=-1))
        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused


class UtteranceAudioTextFusionModel(nn.Module):
    """
    Utterance-level fusion over already-computed modality embeddings.

    Encoders are intentionally optional here so the fusion layer can be tested
    and migrated independently from WavLM/ECAPA and LUAR.
    """

    def __init__(
        self,
        *,
        audio_encoder: AudioEncoder | None = None,
        text_encoder: TextEncoder | None = None,
        fusion: ProjectedConcatFusion,
        embedding_dim: int | None = None,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.embedding_dim = int(embedding_dim if embedding_dim is not None else self.fusion.output.out_features)
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
        text_embeddings: torch.Tensor | None = None,
        pitch_features: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        transcripts: list[str] | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features: dict[str, torch.Tensor] = {}
        if "audio" in self.fusion.input_dims:
            if audio_embeddings is None:
                if self.audio_encoder is None or waveforms is None:
                    raise ValueError("Provide audio_embeddings or attach audio_encoder and waveforms.")
                audio_embeddings = self.audio_encoder(waveforms, padding_mask=padding_mask)
            features["audio"] = audio_embeddings
        if "text" in self.fusion.input_dims:
            if text_embeddings is None:
                if self.text_encoder is None or transcripts is None:
                    raise ValueError("Provide text_embeddings or attach text_encoder and transcripts.")
                encoded = self.text_encoder(transcripts)
                text_embeddings = encoded[0] if isinstance(encoded, tuple) else encoded
            features["text"] = text_embeddings
        if "pitch" in self.fusion.input_dims:
            if pitch_features is None:
                raise ValueError("Provide pitch_features for pitch fusion.")
            if audio_embeddings is None:
                raise ValueError("Audio embeddings are required to infer pitch feature dtype/device.")
            features["pitch"] = pitch_features.to(audio_embeddings.device, dtype=audio_embeddings.dtype)
        return self.fusion(features)


def build_utterance_audio_text_fusion_model(config: ExperimentConfig) -> UtteranceAudioTextFusionModel:
    """Build utterance-level fusion from the released training config schema."""
    options = dict(config.model.options)
    fusion_options = dict(options.pop("fusion", {}) or {})
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
    modalities = tuple(fusion_options.pop("modalities", ("audio", "text")))
    input_dims = {
        name: int(fusion_options.pop(f"{name}_dim"))
        for name in modalities
    }
    fusion = ProjectedConcatFusion(
        input_dims=input_dims,
        projection_dim=int(fusion_options.pop("projection_dim", 256)),
        final_dim=int(fusion_options.pop("final_dim", 192)),
        pre_concat_layernorm=bool(fusion_options.pop("pre_concat_layernorm", False)),
        normalize_pre_concat=bool(fusion_options.pop("normalize_pre_concat", False)),
        normalize=bool(fusion_options.pop("normalize", True)),
    )
    return UtteranceAudioTextFusionModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        fusion=fusion,
        embedding_dim=fusion.output.out_features,
        num_classes=_optional_int(options.pop("num_classes", None)),
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "ProjectedConcatFusion",
    "UtteranceAudioTextFusionModel",
    "build_utterance_audio_text_fusion_model",
]
