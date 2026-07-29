"""Audio encoder builders for WavLM frontends and ECAPA speaker backends."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from multimodal_asv.utils.assets import resolve_asset_path
from multimodal_asv.vendor.ecapa_tdnn import ECAPA_TDNN
from multimodal_asv.vendor.wavlm import WavLM, WavLMConfig


class ECAPABackend(nn.Module):
    """ECAPA-TDNN backend that consumes WavLM frame features."""

    def __init__(
        self,
        *,
        input_dim: int,
        channels: int = 1024,
        pooling_mode: str = "attentive_stats",
        minlstm_hidden_size: int = 3072,
        minlstm_forget_bias_init: float | None = None,
        minlstm_candidate_log_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = ECAPA_TDNN(
            C=channels,
            input_dim=input_dim,
            use_melspec=False,
            pooling_mode=pooling_mode,
            minlstm_hidden_size=minlstm_hidden_size,
            minlstm_forget_bias_init=minlstm_forget_bias_init,
            minlstm_candidate_log_shift=minlstm_candidate_log_shift,
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_frame_features: bool = False,
    ) -> torch.Tensor:
        x = features.transpose(1, 2).contiguous()
        if padding_mask is not None:
            x = x * (~padding_mask).float().unsqueeze(1)
        return self.model(x, aug=False, return_frame_features=return_frame_features)

    def pool_frame_features(
        self,
        frame_features: torch.Tensor,
        frame_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = frame_features
        if frame_padding_mask is not None:
            x = x * (~frame_padding_mask).float().unsqueeze(1)
        return self.model.pool_from_frame_features(x)


class WavLMEcapaEncoder(nn.Module):
    """WavLM frontend followed by an ECAPA-TDNN speaker backend."""

    def __init__(
        self,
        *,
        wavlm_checkpoint: str | None = None,
        wavlm_checkpoint_url: str | None = None,
        download_root: str | None = None,
        allow_download: bool = False,
        freeze_wavlm: bool = True,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.wavlm = self._load_wavlm(
            wavlm_checkpoint,
            wavlm_checkpoint_url=wavlm_checkpoint_url,
            download_root=download_root,
            allow_download=allow_download,
        )
        self.freeze_wavlm = bool(freeze_wavlm)
        if self.freeze_wavlm:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad = False

        feature_dim = int(self.wavlm.cfg.encoder_embed_dim)
        self.backend = ECAPABackend(input_dim=feature_dim, **(backend_kwargs or {}))
        self.last_feature_mask: torch.Tensor | None = None

    def forward(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        return_frame_features: bool = False,
    ) -> torch.Tensor:
        """Encode waveforms into speaker embeddings or frame-level features."""
        if self.freeze_wavlm:
            with torch.no_grad():
                features, feature_mask = self.wavlm.extract_features(
                    waveforms,
                    padding_mask=padding_mask,
                    mask=False,
                )
            features = features.detach()
        else:
            features, feature_mask = self.wavlm.extract_features(
                waveforms,
                padding_mask=padding_mask,
                mask=False,
            )

        self.last_feature_mask = feature_mask
        return self.backend(
            features,
            feature_mask,
            return_frame_features=return_frame_features,
        )

    @staticmethod
    def _load_wavlm(
        checkpoint: str | None,
        *,
        wavlm_checkpoint_url: str | None = None,
        download_root: str | None = None,
        allow_download: bool = False,
    ) -> WavLM:
        """Load WavLM from a local path or an explicit download URL override."""
        resolved = resolve_asset_path(
            checkpoint,
            allow_download=allow_download,
            download_url=wavlm_checkpoint_url,
            download_root=download_root,
            default_filename="wavlm_large.pt",
            asset_name="WavLM checkpoint",
        )
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(
                "WavLM checkpoint not found. Set model.audio_encoder.wavlm_checkpoint "
                "to a local WavLM-Large.pt file."
            )
        state = torch.load(str(resolved), map_location="cpu")
        model = WavLM(WavLMConfig(state["cfg"]))
        model.load_state_dict(state["model"])
        return model


def build_wavlm_ecapa_encoder(config: dict[str, Any]) -> WavLMEcapaEncoder:
    """Build the public WavLM-ECAPA encoder from a config dictionary."""
    return WavLMEcapaEncoder(
        wavlm_checkpoint=config.get("wavlm_checkpoint"),
        wavlm_checkpoint_url=config.get("wavlm_checkpoint_url"),
        download_root=config.get("download_root"),
        allow_download=bool(config.get("allow_download", False)),
        freeze_wavlm=bool(config.get("freeze_wavlm", True)),
        backend_kwargs=dict(config.get("backend_kwargs", {}) or {}),
    )


__all__ = ["ECAPABackend", "WavLMEcapaEncoder", "build_wavlm_ecapa_encoder"]
