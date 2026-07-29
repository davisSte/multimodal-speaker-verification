from __future__ import annotations

import torch
import torchaudio
from torch import nn

from multimodal_asv.config import load_config
from multimodal_asv.data import build_data_pipeline
from multimodal_asv.encoders import IdentityAudioEncoder, StaticTextEncoder
from multimodal_asv.models import (
    FramePooledAudioModel,
    FramePooledAudioTextModel,
    HybridAudioTextModel,
    MeanStdFramePooler,
    QueryAttentionAudioModel,
    ProjectedConcatFusion,
    UtteranceAudioTextFusionModel,
    WeightedSumFusion,
)
from multimodal_asv.training import Trainer


class TinyGroupedAudioEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.proj = nn.Linear(1, self.embedding_dim)

    def forward(self, waveforms: torch.Tensor, *, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        del padding_mask
        if waveforms.dim() != 2:
            raise ValueError(f"Expected flat waveforms, got {tuple(waveforms.shape)}")
        mean = waveforms.mean(dim=1, keepdim=True)
        return self.proj(mean)


class TinyFrameAudioEncoder(nn.Module):
    def __init__(self, frame_dim: int, frames: int) -> None:
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.frames = int(frames)
        self.proj = nn.Linear(1, self.frame_dim)

    def forward(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        return_frame_features: bool = False,
    ) -> torch.Tensor:
        del padding_mask
        if waveforms.dim() != 2:
            raise ValueError(f"Expected flat waveforms, got {tuple(waveforms.shape)}")
        mean = waveforms.mean(dim=1, keepdim=True)
        frame = self.proj(mean).unsqueeze(-1).expand(-1, -1, self.frames)
        if return_frame_features:
            return frame
        return frame.mean(dim=-1)


class TinyTokenizer:
    def __init__(self) -> None:
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.bos_token_id = None
        self.eos_token_id = None
        self.vocab = {
            "<s>": 0,
            "</s>": 1,
            "<pad>": 2,
            "transcript_spk1_0": 3,
            "transcript_spk1_1": 4,
            "transcript_spk2_0": 5,
            "transcript_spk2_1": 6,
        }

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        del padding, truncation, return_tensors
        rows = []
        max_len = 0
        for text in texts:
            tokens = [self.cls_token_id, self.vocab.get(text, 3), self.sep_token_id]
            rows.append(tokens)
            max_len = max(max_len, len(tokens))
        input_ids = []
        attention_mask = []
        for tokens in rows:
            pad = [self.pad_token_id] * (max_len - len(tokens))
            input_ids.append(tokens + pad)
            attention_mask.append([1] * len(tokens) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TinyHybridTextEncoder(nn.Module):
    def __init__(self, token_dim: int = 8, utterance_dim: int = 6) -> None:
        super().__init__()
        self.tokenizer = TinyTokenizer()
        self.token_embeddings = nn.Embedding(7, token_dim)
        self.utterance_proj = nn.Linear(token_dim, utterance_dim)

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embeddings

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(texts)
        input_ids = enc["input_ids"].to(self.token_embeddings.weight.device)
        token_emb = self.token_embeddings(input_ids)
        return self.utterance_proj(token_emb.mean(dim=1))


def _write_wav(path, samples: int, sample_rate: int = 16000) -> None:
    waveform = torch.zeros(1, samples)
    torchaudio.save(str(path), waveform, sample_rate)


def test_trainer_fits_query_attention_audio_and_saves_checkpoint(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path}\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("".join(rows), encoding="utf-8")
    output_dir = tmp_path / "outputs"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_train
model:
  family: query_attention_audio
  embedding_dim: 8
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  speakers_per_batch: 2
  utterances_per_speaker: 2
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  grad_clip: 1.0
  log_interval: 1
output_dir: {output_dir}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = build_data_pipeline(config)
    model = QueryAttentionAudioModel(
        audio_encoder=TinyGroupedAudioEncoder(embedding_dim=8),
        embedding_dim=8,
        num_heads=2,
        dropout=0.0,
        num_classes=2,
    )

    history = Trainer(config).fit(model, pipeline.dataloader)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert (output_dir / "epoch_001.pt").exists()


def test_trainer_fits_frame_pooled_audio_text_and_saves_checkpoint(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_audio_text.txt"
    manifest.write_text("".join(rows), encoding="utf-8")
    output_dir = tmp_path / "outputs_audio_text"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_train_audio_text
model:
  family: frame_pooled_audio_text
  embedding_dim: 5
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  speakers_per_batch: 2
  utterances_per_speaker: 2
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  grad_clip: 1.0
  log_interval: 1
output_dir: {output_dir}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = build_data_pipeline(config)
    model = FramePooledAudioTextModel(
        audio_model=FramePooledAudioModel(
            audio_encoder=TinyFrameAudioEncoder(frame_dim=4, frames=5),
            frame_pooler=MeanStdFramePooler(frame_dim=4, embedding_dim=6),
            embedding_dim=6,
        ),
        text_encoder=StaticTextEncoder(embedding_dim=8),
        fusion=WeightedSumFusion(
            audio_dim=6,
            text_dim=8,
            final_dim=5,
            audio_weight=0.5,
            text_weight=0.5,
        ),
        embedding_dim=5,
        num_classes=2,
    )

    history = Trainer(config).fit(model, pipeline.dataloader)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert (output_dir / "epoch_001.pt").exists()


def test_trainer_fits_utterance_audio_text_and_saves_checkpoint(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_audio_text_flat.txt"
    manifest.write_text("".join(rows), encoding="utf-8")
    output_dir = tmp_path / "outputs_utterance_audio_text"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_train_utterance_audio_text
model:
  family: utterance_audio_text_fusion
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  grad_clip: 1.0
  log_interval: 1
output_dir: {output_dir}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = build_data_pipeline(config)
    model = UtteranceAudioTextFusionModel(
        audio_encoder=IdentityAudioEncoder(),
        text_encoder=StaticTextEncoder(embedding_dim=8),
        fusion=ProjectedConcatFusion(
            input_dims={"audio": 1600, "text": 8},
            projection_dim=16,
            final_dim=5,
        ),
        embedding_dim=5,
        num_classes=2,
    )

    history = Trainer(config).fit(model, pipeline.dataloader)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert (output_dir / "epoch_001.pt").exists()


def test_trainer_fits_hybrid_audio_text_and_saves_checkpoint(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_hybrid_audio_text.txt"
    manifest.write_text("".join(rows), encoding="utf-8")
    output_dir = tmp_path / "outputs_hybrid_audio_text"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_train_hybrid_audio_text
model:
  family: hybrid_audio_text
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  grad_clip: 1.0
  log_interval: 1
output_dir: {output_dir}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    pipeline = build_data_pipeline(config)
    model = HybridAudioTextModel(
        audio_encoder=TinyFrameAudioEncoder(frame_dim=4, frames=5),
        text_encoder=TinyHybridTextEncoder(token_dim=8, utterance_dim=6),
        frame_dim=4,
        d_k=8,
        d_v=8,
        hybrid_dim=8,
        final_dim=5,
        text_dim=8,
        sluar_dim=6,
        num_classes=2,
    )

    history = Trainer(config).fit(model, pipeline.dataloader)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert (output_dir / "epoch_001.pt").exists()
