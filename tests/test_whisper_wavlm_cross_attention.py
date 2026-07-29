from __future__ import annotations

import types

import torch
import torch.nn as nn

from multimodal_asv.config import load_config
from multimodal_asv.models import WhisperWavlmCrossAttentionModel
from multimodal_asv.registry import build_model
import multimodal_asv.models.whisper_wavlm_cross_attention as wwca


class DummyWhisperEncoder(nn.Module):
    def __init__(self, n_audio_state: int) -> None:
        super().__init__()
        self.n_audio_state = n_audio_state

    def forward(self, mel: torch.Tensor, layer_idx=None) -> torch.Tensor:
        del layer_idx
        batch_size, _, n_frames = mel.shape
        out_len = max(1, n_frames // 2)
        return torch.zeros(batch_size, out_len, self.n_audio_state, device=mel.device, dtype=mel.dtype)


class DummyWhisper(nn.Module):
    def __init__(self, n_mels: int = 4, n_audio_state: int = 8) -> None:
        super().__init__()
        self.dims = types.SimpleNamespace(n_audio_state=n_audio_state, n_mels=n_mels)
        self.encoder = DummyWhisperEncoder(n_audio_state)


class DummyAudioEncoder(nn.Module):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.last_feature_mask = None

    def forward(self, waveforms, padding_mask=None, return_frame_features=False):
        del padding_mask
        batch_size = waveforms.size(0)
        if return_frame_features:
            self.last_feature_mask = torch.zeros(batch_size, 5, dtype=torch.bool, device=waveforms.device)
            return torch.zeros(batch_size, 6, 5, device=waveforms.device, dtype=waveforms.dtype)
        return torch.zeros(batch_size, 192, device=waveforms.device, dtype=waveforms.dtype)


class DummyWhisperAudio:
    N_SAMPLES = 16
    HOP_LENGTH = 2
    N_FRAMES = 8

    @staticmethod
    def pad_or_trim(array, length: int, axis: int = -1):
        axis = axis if axis >= 0 else array.ndim + axis
        if array.shape[axis] > length:
            return array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if array.shape[axis] < length:
            pad = [0, 0] * array.ndim
            pad_index = (array.ndim - 1 - axis) * 2
            pad[pad_index] = length - array.shape[axis]
            return torch.nn.functional.pad(array, pad)
        return array

    @staticmethod
    def log_mel_spectrogram(audio, n_mels: int = 4, padding: int = 0, device=None):
        del padding
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio)
        if device is not None:
            audio = audio.to(device)
        if audio.dim() == 1:
            return torch.zeros(n_mels, DummyWhisperAudio.N_FRAMES, device=audio.device)
        return torch.zeros(audio.size(0), n_mels, DummyWhisperAudio.N_FRAMES, device=audio.device)


def _patch_minimal(monkeypatch) -> None:
    monkeypatch.setattr(wwca, "_load_whisper_model", lambda *_args, **_kwargs: DummyWhisper())
    monkeypatch.setattr(wwca, "_load_whisper_modules", lambda: (DummyWhisperAudio, types.SimpleNamespace()))


def test_whisper_wavlm_cross_attention_output_shape(monkeypatch) -> None:
    _patch_minimal(monkeypatch)
    model = WhisperWavlmCrossAttentionModel(
        audio_encoder=DummyAudioEncoder(),
        whisper_encoder=wwca.WhisperSequenceEncoder(checkpoint="dummy"),
        whisper_fusion_type="cross_attn",
        whisper_attn_window_k=0,
        frame_dim=6,
        whisper_dim=8,
        d_k=4,
        d_v=8,
        cross_attn_heads=2,
        embedding_dim=6,
        normalize=True,
        pos_max_len=32,
        whisper_pre_concat_layernorm=False,
        whisper_grouped=False,
        whisper_concat_fullseq=False,
        whisper_pos_offsets=True,
        use_self_attn=True,
    )
    waveforms = torch.zeros(2, 10)
    padding_mask = torch.zeros(2, 10, dtype=torch.bool)
    out = model(waveforms, padding_mask=padding_mask)
    assert out.shape == (2, 6)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_whisper_wavlm_cross_attention_proj_output_shape(monkeypatch) -> None:
    _patch_minimal(monkeypatch)
    model = WhisperWavlmCrossAttentionModel(
        audio_encoder=DummyAudioEncoder(),
        whisper_encoder=wwca.WhisperSequenceEncoder(checkpoint="dummy"),
        whisper_fusion_type="cross_attn",
        frame_dim=6,
        whisper_dim=8,
        d_k=4,
        d_v=4,
        cross_attn_heads=2,
        embedding_dim=5,
        normalize=False,
        pos_max_len=32,
        whisper_grouped=False,
        whisper_concat_fullseq=False,
        whisper_pos_offsets=True,
        use_self_attn=True,
    )
    out = model(torch.zeros(1, 12))
    assert out.shape == (1, 5)


def test_whisper_wavlm_interp_proj_concat_output_shape(monkeypatch) -> None:
    _patch_minimal(monkeypatch)
    model = WhisperWavlmCrossAttentionModel(
        audio_encoder=DummyAudioEncoder(),
        whisper_encoder=wwca.WhisperSequenceEncoder(checkpoint="dummy"),
        whisper_fusion_type="interp_proj_concat",
        frame_dim=6,
        whisper_dim=8,
        d_k=4,
        d_v=4,
        cross_attn_heads=2,
        embedding_dim=5,
        normalize=False,
        pooling_mode="attentive_stats",
        pos_max_len=32,
        whisper_grouped=False,
        whisper_concat_fullseq=False,
        whisper_pos_offsets=True,
        use_self_attn=True,
    )
    out = model(torch.zeros(2, 12))
    assert out.shape == (2, 5)


def test_whisper_wavlm_windowed_attention_mask(monkeypatch) -> None:
    _patch_minimal(monkeypatch)
    model = WhisperWavlmCrossAttentionModel(
        audio_encoder=DummyAudioEncoder(),
        whisper_encoder=wwca.WhisperSequenceEncoder(checkpoint="dummy"),
        whisper_fusion_type="interp_windowed_cross_attn",
        whisper_attn_window_k=1,
        frame_dim=6,
        whisper_dim=8,
        d_k=4,
        d_v=4,
        cross_attn_heads=2,
        embedding_dim=5,
        normalize=False,
        pos_max_len=32,
        whisper_grouped=False,
        whisper_concat_fullseq=False,
        whisper_pos_offsets=True,
        use_self_attn=False,
    )
    audio_seq = torch.zeros(1, 5, 4)
    audio_mask = torch.zeros(1, 5, dtype=torch.bool)
    whisper_mask = torch.zeros(1, 5, dtype=torch.bool)
    attn_mask = model._build_local_cross_attn_mask(audio_seq, audio_mask, whisper_mask, window_k=1)
    assert attn_mask.shape == (1, 5, 5)
    assert bool(attn_mask[0, 2, 0].item()) is True
    assert bool(attn_mask[0, 2, 1].item()) is False
    assert bool(attn_mask[0, 2, 2].item()) is False
    assert bool(attn_mask[0, 2, 3].item()) is False
    assert bool(attn_mask[0, 2, 4].item()) is True


def test_whisper_wavlm_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_whisper_wavlm_cross_attention
model:
  family: whisper_wavlm_cross_attention
  embedding_dim: 5
  fusion:
    type: cross_attn
    pooling: mean
  hybrid:
    frame_dim: 6
    whisper_dim: 8
    d_k: 4
    d_v: 4
    cross_attn_heads: 2
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, WhisperWavlmCrossAttentionModel)
