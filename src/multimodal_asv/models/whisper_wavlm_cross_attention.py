"""Whisper-WavLM fusion models with cross-attention over sequence features."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig


def _load_whisper_modules() -> tuple[object, object]:
    try:
        audio_module = importlib.import_module("whisper.audio")
        model_module = importlib.import_module("whisper.model")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Whisper support requires the optional dependency. Install it with "
            "`uv sync --group dev --extra whisper`."
        ) from exc
    return audio_module, model_module


def _resolve_whisper_checkpoint(name_or_path: str, download_root: str | None) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    root_path = Path(download_root) if download_root is not None else Path.home() / ".cache" / "whisper"
    file_path = root_path / f"{name_or_path}.pt"
    if file_path.is_file():
        return file_path
    raise RuntimeError(
        f"Whisper checkpoint not found for {name_or_path!r}. "
        f"Provide a valid checkpoint path or place the model at {file_path}."
    )


def _load_whisper_model(
    checkpoint: str,
    device: torch.device,
    download_root: str | None,
    allow_download: bool,
) -> nn.Module:
    try:
        whisper_pkg = importlib.import_module("whisper")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Whisper support requires the optional dependency. Install it with "
            "`uv sync --group dev --extra whisper`."
        ) from exc
    if allow_download:
        return whisper_pkg.load_model(
            checkpoint,
            device=device,
            download_root=download_root,
            in_memory=False,
        )
    checkpoint_path = _resolve_whisper_checkpoint(checkpoint, download_root)
    return whisper_pkg.load_model(
        str(checkpoint_path),
        device=device,
        download_root=download_root,
        in_memory=False,
    )


class WhisperSequenceEncoder(nn.Module):
    """Whisper encoder wrapper that returns sequence states and a padding mask."""

    def __init__(
        self,
        *,
        checkpoint: str,
        download_root: str | None = None,
        allow_download: bool = False,
        freeze_model: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.model = _load_whisper_model(
            checkpoint,
            device=torch.device("cpu"),
            download_root=download_root,
            allow_download=allow_download,
        )
        self.freeze_model = bool(freeze_model)
        if self.freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.layer_idx = layer_idx

    @property
    def hidden_size(self) -> int:
        return int(self.model.dims.n_audio_state)

    def forward(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        whisper_audio, _ = _load_whisper_modules()
        device = waveforms.device
        if padding_mask is None:
            lengths = torch.full(
                (waveforms.size(0),),
                waveforms.size(1),
                dtype=torch.long,
                device=device,
            )
        else:
            lengths = (~padding_mask).sum(dim=1).long()

        padded = whisper_audio.pad_or_trim(waveforms, length=whisper_audio.N_SAMPLES, axis=-1)
        mel = whisper_audio.log_mel_spectrogram(padded, n_mels=self.model.dims.n_mels, device=device)
        if self.freeze_model:
            with torch.no_grad():
                seq = self.model.encoder(mel, layer_idx=self.layer_idx)
        else:
            seq = self.model.encoder(mel, layer_idx=self.layer_idx)

        max_enc_len = seq.size(1)
        mel_len = (lengths // whisper_audio.HOP_LENGTH) + 1
        mel_len = torch.clamp(mel_len, min=1, max=whisper_audio.N_FRAMES)
        enc_len = (mel_len + 1) // 2
        enc_len = torch.clamp(enc_len, min=1, max=max_enc_len)
        trunc_len = int(enc_len.max().item())
        seq = seq[:, :trunc_len, :]
        pad = torch.arange(trunc_len, device=device).unsqueeze(0) >= enc_len.unsqueeze(1)
        return seq, pad


class CrossAttentionAligner(nn.Module):
    """Cross-attention from audio queries to Whisper keys/values."""

    def __init__(
        self,
        *,
        audio_dim: int,
        whisper_dim: int,
        d_k: int,
        d_v: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if d_k % num_heads != 0:
            raise ValueError("d_k must be divisible by num_heads.")
        if d_v % num_heads != 0:
            raise ValueError("d_v must be divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.head_dk = self.d_k // self.num_heads
        self.head_dv = self.d_v // self.num_heads
        self.q_proj = nn.Linear(audio_dim, self.d_k)
        self.k_proj = nn.Linear(whisper_dim, self.d_k)
        self.v_proj = nn.Linear(whisper_dim, self.d_v)
        self.out_proj = nn.Linear(self.d_v, self.d_v)
        self.last_alpha: torch.Tensor | None = None

    def forward(
        self,
        audio_seq: torch.Tensor,
        whisper_seq: torch.Tensor,
        *,
        audio_padding_mask: torch.Tensor | None = None,
        whisper_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, audio_len, _ = audio_seq.shape
        whisper_len = whisper_seq.size(1)

        q = self.q_proj(audio_seq).view(batch, audio_len, self.num_heads, self.head_dk).transpose(1, 2)
        k = self.k_proj(whisper_seq).view(batch, whisper_len, self.num_heads, self.head_dk).transpose(1, 2)
        v = self.v_proj(whisper_seq).view(batch, whisper_len, self.num_heads, self.head_dv).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dk ** 0.5)
        block_mask = torch.zeros_like(scores, dtype=torch.bool)
        if audio_padding_mask is not None:
            block_mask |= audio_padding_mask.unsqueeze(1).unsqueeze(-1)
        if whisper_padding_mask is not None:
            block_mask |= whisper_padding_mask.unsqueeze(1).unsqueeze(2)
        if attn_mask is not None:
            block_mask |= attn_mask.unsqueeze(1)
        scores = scores.masked_fill(block_mask, -1e9)
        alpha = F.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(block_mask, 0.0)
        self.last_alpha = alpha.detach()
        out = torch.matmul(alpha, v)
        out = out.transpose(1, 2).contiguous().view(batch, audio_len, self.d_v)
        return self.out_proj(out)


class WhisperWavlmCrossAttentionModel(nn.Module):
    """Whisper/WavLM sequence fusion with cross-attention or interpolation fusion."""

    def __init__(
        self,
        *,
        audio_encoder: nn.Module | None = None,
        whisper_encoder: WhisperSequenceEncoder | None = None,
        whisper_fusion_type: str = "cross_attn",
        whisper_attn_window_k: int = 0,
        pos_max_len: int = 1200,
        whisper_pre_concat_layernorm: bool = False,
        whisper_grouped: bool = True,
        whisper_concat_fullseq: bool = True,
        whisper_pos_offsets: bool = True,
        use_self_attn: bool = True,
        frame_dim: int = 1536,
        whisper_dim: int = 1024,
        d_k: int = 256,
        d_v: int = 256,
        cross_attn_heads: int = 4,
        embedding_dim: int = 192,
        normalize: bool = True,
        pooling_mode: str = "attentive_stats",
        pooling_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.whisper_encoder = whisper_encoder
        self.whisper_fusion_type = str(whisper_fusion_type)
        self.whisper_attn_window_k = int(whisper_attn_window_k)
        self.max_len = int(pos_max_len)
        self.whisper_grouped = bool(whisper_grouped)
        self.whisper_concat_fullseq = bool(whisper_concat_fullseq)
        self.whisper_pos_offsets = bool(whisper_pos_offsets)
        self.use_self_attn = bool(use_self_attn)
        self.normalize = bool(normalize)
        self.pooling_mode = str(pooling_mode)
        self.last_pool_weights: torch.Tensor | None = None

        if self.whisper_fusion_type not in {"cross_attn", "interp_proj_concat", "interp_windowed_cross_attn"}:
            raise ValueError(
                "whisper_fusion_type must be one of "
                "['cross_attn', 'interp_proj_concat', 'interp_windowed_cross_attn']."
            )

        self.fusion_dim = int(d_k)
        self.pool_input_dim = self.fusion_dim
        self.pos_audio = nn.Parameter(torch.randn(1, self.max_len, self.fusion_dim))
        self.pos_whisper = nn.Parameter(torch.randn(1, self.max_len, self.fusion_dim))

        if self.whisper_fusion_type in {"cross_attn", "interp_windowed_cross_attn"}:
            self.audio_proj = nn.Linear(frame_dim, self.fusion_dim)
            self.whisper_proj = nn.Linear(whisper_dim, self.fusion_dim)
            self.cross_attention = CrossAttentionAligner(
                audio_dim=self.fusion_dim,
                whisper_dim=self.fusion_dim,
                d_k=d_k,
                d_v=d_v,
                num_heads=cross_attn_heads,
            )
            self.self_attn = nn.MultiheadAttention(self.fusion_dim, cross_attn_heads, batch_first=True)
            self.self_attn_ln = nn.LayerNorm(self.fusion_dim)
            self.cross_ln_audio = nn.LayerNorm(self.fusion_dim)
            self.cross_ln_whisper = nn.LayerNorm(self.fusion_dim)
            self.attn_resid_proj = nn.Identity() if d_v == self.fusion_dim else nn.Linear(d_v, self.fusion_dim)
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.fusion_dim),
                nn.Linear(self.fusion_dim, self.fusion_dim * 4),
                nn.GELU(),
                nn.Linear(self.fusion_dim * 4, self.fusion_dim),
            )
            self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        else:
            self.interp_audio_proj = nn.Linear(frame_dim, self.fusion_dim)
            self.interp_whisper_proj = nn.Linear(whisper_dim, self.fusion_dim)
            self.interp_audio_ln = nn.LayerNorm(self.fusion_dim) if whisper_pre_concat_layernorm else nn.Identity()
            self.interp_whisper_ln = nn.LayerNorm(self.fusion_dim) if whisper_pre_concat_layernorm else nn.Identity()
            self.frame_fusion_proj = nn.Linear(self.fusion_dim * 2, embedding_dim)
            self.pool_input_dim = embedding_dim

        pool_out_dim = self.pool_input_dim * 2 if self.pooling_mode == "attentive_stats" else self.pool_input_dim
        self.embedding_proj = nn.Identity() if embedding_dim == pool_out_dim else nn.Linear(pool_out_dim, embedding_dim)
        self.attn_pool = None
        if self.pooling_mode == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(self.pool_input_dim, pooling_hidden_dim),
                nn.Tanh(),
                nn.Linear(pooling_hidden_dim, 1),
            )
        self.attn_stats = None
        if self.pooling_mode == "attentive_stats":
            self.attn_stats = nn.Sequential(
                nn.Conv1d(self.pool_input_dim * 3, pooling_hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(pooling_hidden_dim),
                nn.Tanh(),
                nn.Conv1d(pooling_hidden_dim, self.pool_input_dim, kernel_size=1),
            )

    def forward(
        self,
        waveforms: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        audio_seq: torch.Tensor | None = None,
        whisper_seq: torch.Tensor | None = None,
        audio_padding_mask: torch.Tensor | None = None,
        whisper_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_seq is None or whisper_seq is None:
            if waveforms is None:
                raise ValueError("Provide waveforms or both audio_seq and whisper_seq.")
            audio_seq, whisper_seq, audio_padding_mask, whisper_padding_mask = self._encode_inputs(
                waveforms,
                padding_mask=padding_mask,
            )

        if waveforms is not None and waveforms.dim() == 3:
            batch_size, num_utts, _ = waveforms.shape
            if self.whisper_concat_fullseq:
                audio_seq, audio_padding_mask, whisper_seq, whisper_padding_mask = self._concat_grouped_sequences(
                    audio_seq,
                    whisper_seq,
                    audio_padding_mask,
                    whisper_padding_mask,
                    batch_size=batch_size,
                    num_utts=num_utts,
                )
                fused_seq, fused_mask = self._fuse_single(
                    audio_seq,
                    whisper_seq,
                    audio_padding_mask,
                    whisper_padding_mask,
                    0,
                    0,
                    positions_already_added=True,
                )
            else:
                fused_batches = []
                fused_masks = []
                for b in range(batch_size):
                    utterance_fused = []
                    utterance_masks = []
                    audio_offset = 0
                    whisper_offset = 0
                    for u in range(num_utts):
                        idx = b * num_utts + u
                        fused_u, mask_u = self._fuse_single(
                            audio_seq[idx : idx + 1],
                            whisper_seq[idx : idx + 1],
                            audio_padding_mask[idx : idx + 1] if audio_padding_mask is not None else None,
                            whisper_padding_mask[idx : idx + 1] if whisper_padding_mask is not None else None,
                            audio_offset,
                            whisper_offset,
                        )
                        utterance_fused.append(fused_u)
                        utterance_masks.append(mask_u)
                        if self.whisper_pos_offsets:
                            audio_offset += fused_u.size(1)
                            whisper_offset += whisper_seq[idx : idx + 1].size(1)
                    fused_batches.append(torch.cat(utterance_fused, dim=1))
                    fused_masks.append(torch.cat(utterance_masks, dim=1))
                fused_seq = torch.cat(fused_batches, dim=0)
                fused_mask = torch.cat(fused_masks, dim=0)
        else:
            fused_seq, fused_mask = self._fuse_single(
                audio_seq,
                whisper_seq,
                audio_padding_mask,
                whisper_padding_mask,
                0,
                0,
            )

        pooled = self._pool_sequence(fused_seq, fused_mask)
        emb = self.embedding_proj(pooled)
        if self.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb

    def _encode_inputs(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.audio_encoder is None or self.whisper_encoder is None:
            raise ValueError("Waveform encoding requires both audio_encoder and whisper_encoder.")

        if waveforms.dim() == 3:
            batch_size, num_utts, num_samples = waveforms.shape
            flat_waveforms = waveforms.reshape(batch_size * num_utts, num_samples)
            flat_padding_mask = padding_mask.reshape(batch_size * num_utts, num_samples) if padding_mask is not None else None
        elif waveforms.dim() == 2:
            flat_waveforms = waveforms
            flat_padding_mask = padding_mask
        else:
            raise ValueError(f"Expected waveforms to be (B, T) or (B, N, T), got {tuple(waveforms.shape)}")

        whisper_seq, whisper_pad = self.whisper_encoder(flat_waveforms, padding_mask=flat_padding_mask)
        frame_map = self.audio_encoder(
            flat_waveforms,
            padding_mask=flat_padding_mask,
            return_frame_features=True,
        )
        audio_pad = self.audio_encoder.last_feature_mask
        audio_pad = audio_pad.to(torch.bool) if audio_pad is not None else None
        audio_seq = frame_map.transpose(1, 2)
        return audio_seq, whisper_seq, audio_pad, whisper_pad

    def _fuse_single(
        self,
        audio_seq: torch.Tensor,
        whisper_seq: torch.Tensor,
        audio_mask: torch.Tensor | None,
        whisper_mask: torch.Tensor | None,
        audio_offset: int,
        whisper_offset: int,
        positions_already_added: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.whisper_fusion_type == "interp_proj_concat":
            return self._fuse_interp_proj_concat(audio_seq, whisper_seq, audio_mask, whisper_mask)
        if self.whisper_fusion_type == "interp_windowed_cross_attn":
            return self._fuse_interp_windowed_cross_attn(
                audio_seq,
                whisper_seq,
                audio_mask,
                whisper_mask,
                audio_offset,
                whisper_offset,
                positions_already_added=positions_already_added,
            )

        target_len = audio_seq.size(1)
        if audio_mask is None:
            audio_mask = torch.zeros((audio_seq.size(0), target_len), dtype=torch.bool, device=audio_seq.device)
        audio_proj = self.audio_proj(audio_seq)
        whisper_proj = self.whisper_proj(whisper_seq)
        if not positions_already_added:
            audio_proj = self._add_positions(audio_proj, self.pos_audio, audio_offset, "audio")
            whisper_proj = self._add_positions(whisper_proj, self.pos_whisper, whisper_offset, "whisper")
        if self.use_self_attn:
            self_attn_out, _ = self.self_attn(
                self.self_attn_ln(audio_proj),
                self.self_attn_ln(audio_proj),
                self.self_attn_ln(audio_proj),
                key_padding_mask=audio_mask,
            )
            audio_proj = audio_proj + self_attn_out
        attn_out = self.cross_attention(
            self.cross_ln_audio(audio_proj),
            self.cross_ln_whisper(whisper_proj),
            audio_padding_mask=audio_mask,
            whisper_padding_mask=whisper_mask,
        )
        fused = audio_proj + self.attn_resid_proj(attn_out)
        fused = fused + self.mlp(self.fusion_norm(fused))
        return fused, audio_mask

    def _fuse_interp_proj_concat(
        self,
        audio_seq: torch.Tensor,
        whisper_seq: torch.Tensor,
        audio_mask: torch.Tensor | None,
        whisper_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_whisper, fused_mask = self._align_whisper_to_audio(whisper_seq, audio_seq, whisper_mask, audio_mask)
        audio_proj = self.interp_audio_ln(self.interp_audio_proj(audio_seq))
        whisper_proj = self.interp_whisper_ln(self.interp_whisper_proj(aligned_whisper))
        fused = self.frame_fusion_proj(torch.cat((audio_proj, whisper_proj), dim=-1))
        return fused, fused_mask

    def _fuse_interp_windowed_cross_attn(
        self,
        audio_seq: torch.Tensor,
        whisper_seq: torch.Tensor,
        audio_mask: torch.Tensor | None,
        whisper_mask: torch.Tensor | None,
        audio_offset: int,
        whisper_offset: int,
        positions_already_added: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_whisper, fused_mask = self._align_whisper_to_audio(whisper_seq, audio_seq, whisper_mask, audio_mask)
        if fused_mask is None:
            fused_mask = torch.zeros((audio_seq.size(0), audio_seq.size(1)), dtype=torch.bool, device=audio_seq.device)
        audio_proj = self.audio_proj(audio_seq)
        whisper_proj = self.whisper_proj(aligned_whisper)
        if not positions_already_added:
            audio_proj = self._add_positions(audio_proj, self.pos_audio, audio_offset, "audio")
            whisper_proj = self._add_positions(whisper_proj, self.pos_whisper, whisper_offset, "whisper")
        if self.use_self_attn:
            self_attn_out, _ = self.self_attn(
                self.self_attn_ln(audio_proj),
                self.self_attn_ln(audio_proj),
                self.self_attn_ln(audio_proj),
                key_padding_mask=fused_mask,
            )
            audio_proj = audio_proj + self_attn_out
        local_mask = self._build_local_cross_attn_mask(audio_proj, fused_mask, fused_mask, self.whisper_attn_window_k)
        attn_out = self.cross_attention(
            self.cross_ln_audio(audio_proj),
            self.cross_ln_whisper(whisper_proj),
            audio_padding_mask=fused_mask,
            whisper_padding_mask=fused_mask,
            attn_mask=local_mask,
        )
        fused = audio_proj + self.attn_resid_proj(attn_out)
        fused = fused + self.mlp(self.fusion_norm(fused))
        return fused, fused_mask

    def _pool_sequence(self, seq: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling_mode == "attn":
            return self._masked_attn_pool(seq, pad_mask)
        if self.pooling_mode == "attentive_stats":
            return self._masked_attentive_stats_pool(seq, pad_mask)
        return self._masked_mean(seq, pad_mask)

    def _masked_mean(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if pad_mask is None:
            return x.mean(dim=1)
        valid = (~pad_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def _masked_attn_pool(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if self.attn_pool is None:
            raise RuntimeError("Attention pooling head is not initialized.")
        scores = self.attn_pool(x).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        weights = F.softmax(scores, dim=1)
        self.last_pool_weights = weights.detach()
        return (x * weights.unsqueeze(-1)).sum(dim=1)

    def _masked_attentive_stats_pool(self, x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if self.attn_stats is None:
            raise RuntimeError("Attentive stats pooling head is not initialized.")
        frame_x = x.transpose(1, 2)
        time = frame_x.size(-1)
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(1).float()
            denom = valid.sum(dim=2, keepdim=True) + 1e-6
            mean = (frame_x * valid).sum(dim=2, keepdim=True) / denom
            var = (valid * (frame_x - mean) ** 2).sum(dim=2, keepdim=True) / denom
            std = torch.sqrt(var.clamp(min=1e-4))
        else:
            mean = torch.mean(frame_x, dim=2, keepdim=True)
            std = torch.sqrt(torch.var(frame_x, dim=2, keepdim=True).clamp(min=1e-4))
        global_x = torch.cat((frame_x, mean.repeat(1, 1, time), std.repeat(1, 1, time)), dim=1)
        logits = self.attn_stats(global_x)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask.unsqueeze(1), -1e9)
        weights = F.softmax(logits, dim=2)
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(1).float()
            weights = weights * valid
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        self.last_pool_weights = weights.detach()
        mu = torch.sum(frame_x * weights, dim=2)
        second_moment = torch.sum((frame_x ** 2) * weights, dim=2)
        sg = torch.sqrt((second_moment - mu**2).clamp(min=1e-4))
        return torch.cat((mu, sg), dim=1)

    def _align_whisper_to_audio(
        self,
        whisper_seq: torch.Tensor,
        audio_seq: torch.Tensor,
        whisper_mask: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, max_audio_len, _ = audio_seq.shape
        aligned = audio_seq.new_zeros(batch, max_audio_len, whisper_seq.size(-1))
        if audio_mask is None:
            audio_mask = torch.zeros((batch, max_audio_len), dtype=torch.bool, device=audio_seq.device)
        for idx in range(batch):
            target_len = int((~audio_mask[idx]).sum().item())
            if target_len <= 0:
                continue
            source_len = whisper_seq.size(1) if whisper_mask is None else int((~whisper_mask[idx]).sum().item())
            source_len = max(1, source_len)
            valid = whisper_seq[idx : idx + 1, :source_len, :].transpose(1, 2)
            aligned_valid = F.interpolate(valid, size=target_len, mode="linear", align_corners=False)
            aligned[idx, :target_len, :] = aligned_valid.transpose(1, 2).squeeze(0)
        return aligned, audio_mask

    def _build_local_cross_attn_mask(
        self,
        audio_seq: torch.Tensor,
        audio_mask: torch.Tensor | None,
        whisper_mask: torch.Tensor | None,
        window_k: int,
    ) -> torch.Tensor:
        batch, target_len, _ = audio_seq.shape
        device = audio_seq.device
        idx = torch.arange(target_len, device=device)
        local = (idx.view(-1, 1) - idx.view(1, -1)).abs() > int(window_k)
        attn_mask = local.unsqueeze(0).expand(batch, -1, -1).clone()
        if whisper_mask is not None:
            attn_mask |= whisper_mask.unsqueeze(1).expand(-1, target_len, -1)
        if audio_mask is not None:
            attn_mask |= audio_mask.unsqueeze(2).expand(-1, -1, target_len)
        return attn_mask

    def _concat_grouped_sequences(
        self,
        audio_seq: torch.Tensor,
        whisper_seq: torch.Tensor,
        audio_mask: torch.Tensor | None,
        whisper_mask: torch.Tensor | None,
        *,
        batch_size: int,
        num_utts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_batches: list[torch.Tensor] = []
        audio_masks: list[torch.Tensor] = []
        whisper_batches: list[torch.Tensor] = []
        whisper_masks: list[torch.Tensor] = []
        device = audio_seq.device

        for b in range(batch_size):
            audio_parts: list[torch.Tensor] = []
            audio_mask_parts: list[torch.Tensor] = []
            whisper_parts: list[torch.Tensor] = []
            whisper_mask_parts: list[torch.Tensor] = []
            for u in range(num_utts):
                idx = b * num_utts + u
                a = audio_seq[idx : idx + 1]
                w = whisper_seq[idx : idx + 1]
                a_mask = audio_mask[idx : idx + 1] if audio_mask is not None else None
                w_mask = whisper_mask[idx : idx + 1] if whisper_mask is not None else None
                a_len = int((~a_mask).sum(dim=1).item()) if a_mask is not None else a.size(1)
                w_len = int((~w_mask).sum(dim=1).item()) if w_mask is not None else w.size(1)
                audio_parts.append(self._add_positions(a[:, :a_len], self.pos_audio, 0, "audio"))
                whisper_parts.append(self._add_positions(w[:, :w_len], self.pos_whisper, 0, "whisper"))
                audio_mask_parts.append(torch.zeros((1, a_len), dtype=torch.bool, device=device))
                whisper_mask_parts.append(torch.zeros((1, w_len), dtype=torch.bool, device=device))
            audio_cat = torch.cat(audio_parts, dim=1)
            whisper_cat = torch.cat(whisper_parts, dim=1)
            audio_mask_cat = torch.cat(audio_mask_parts, dim=1)
            whisper_mask_cat = torch.cat(whisper_mask_parts, dim=1)
            audio_batches.append(audio_cat)
            audio_masks.append(audio_mask_cat)
            whisper_batches.append(whisper_cat)
            whisper_masks.append(whisper_mask_cat)

        max_audio_len = max(seq.size(1) for seq in audio_batches)
        max_whisper_len = max(seq.size(1) for seq in whisper_batches)
        audio_cat, audio_mask_out = self._pad_batch(audio_batches, audio_masks, max_audio_len)
        whisper_cat, whisper_mask_out = self._pad_batch(whisper_batches, whisper_masks, max_whisper_len)
        return audio_cat, audio_mask_out, whisper_cat, whisper_mask_out

    @staticmethod
    def _pad_batch(
        seqs: list[torch.Tensor],
        masks: list[torch.Tensor],
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded_seqs: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for seq, mask in zip(seqs, masks):
            if seq.size(1) < max_len:
                pad_len = max_len - seq.size(1)
                seq = F.pad(seq, (0, 0, 0, pad_len))
                mask = F.pad(mask, (0, pad_len), value=True)
            padded_seqs.append(seq)
            padded_masks.append(mask)
        return torch.cat(padded_seqs, dim=0), torch.cat(padded_masks, dim=0)

    def _add_positions(self, seq: torch.Tensor, pos: torch.Tensor, offset: int, label: str) -> torch.Tensor:
        length = seq.size(1)
        if offset + length > self.max_len:
            raise ValueError(f"Increase pos_max_len for {label}: offset={offset} len={length}")
        return seq + pos[:, offset : offset + length, :]


def build_whisper_wavlm_cross_attention_model(config: ExperimentConfig) -> WhisperWavlmCrossAttentionModel:
    options = dict(config.model.options)
    audio_config = options.pop("audio_encoder", None)
    whisper_config = dict(options.pop("whisper_encoder", {}) or {})
    audio_encoder = None
    if audio_config is not None:
        from multimodal_asv.encoders.audio import build_wavlm_ecapa_encoder

        audio_encoder = build_wavlm_ecapa_encoder(audio_config)

    whisper_encoder = None
    if whisper_config:
        whisper_encoder = WhisperSequenceEncoder(
            checkpoint=str(whisper_config.get("checkpoint", whisper_config.get("checkpoint_path", "medium"))),
            download_root=_optional_str(whisper_config.get("download_root")),
            allow_download=bool(whisper_config.get("allow_download", False)),
            freeze_model=bool(whisper_config.get("freeze_model", True)),
            layer_idx=_optional_int(whisper_config.get("layer_idx")),
        )

    fusion = dict(options.pop("fusion", {}) or {})
    hybrid = dict(options.pop("hybrid", {}) or {})
    return WhisperWavlmCrossAttentionModel(
        audio_encoder=audio_encoder,
        whisper_encoder=whisper_encoder,
        whisper_fusion_type=str(fusion.pop("type", "cross_attn")),
        whisper_attn_window_k=int(fusion.pop("attn_window_k", 0)),
        pos_max_len=int(fusion.pop("pos_max_len", 1200)),
        whisper_pre_concat_layernorm=bool(fusion.pop("pre_concat_layernorm", False)),
        whisper_grouped=bool(fusion.pop("grouped", True)),
        whisper_concat_fullseq=bool(fusion.pop("concat_fullseq", True)),
        whisper_pos_offsets=bool(fusion.pop("pos_offsets", True)),
        use_self_attn=bool(fusion.pop("self_attn", True)),
        frame_dim=int(hybrid.pop("frame_dim", 1536)),
        whisper_dim=int(hybrid.pop("whisper_dim", 1024)),
        d_k=int(hybrid.pop("d_k", 256)),
        d_v=int(hybrid.pop("d_v", 256)),
        cross_attn_heads=int(hybrid.pop("cross_attn_heads", 4)),
        embedding_dim=int(options.pop("embedding_dim", 192)),
        normalize=bool(options.pop("normalize", True)),
        pooling_mode=str(fusion.pop("pooling", "attentive_stats")),
        pooling_hidden_dim=int(fusion.pop("pooling_hidden_dim", 128)),
    )


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "CrossAttentionAligner",
    "WhisperSequenceEncoder",
    "WhisperWavlmCrossAttentionModel",
    "build_whisper_wavlm_cross_attention_model",
]
