from __future__ import annotations

import torch
import torchaudio

from multimodal_asv.config import load_config
from multimodal_asv.data import build_data_pipeline


def _write_wav(path, samples: int, sample_rate: int = 16000) -> None:
    waveform = torch.zeros(1, samples)
    torchaudio.save(str(path), waveform, sample_rate)


def test_build_query_attention_audio_data_pipeline(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path}\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_query_attention_audio
model:
  family: query_attention_audio
  embedding_dim: 192
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
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 1
    assert batch["waveforms"].shape == (2, 2, 1600)


def test_build_frame_pooled_audio_data_pipeline(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path}\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_frame_pooled_audio
model:
  family: frame_pooled_audio
  embedding_dim: 192
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
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 1
    assert batch["waveforms"].shape == (2, 2, 1600)


def test_build_frame_pooled_audio_text_data_pipeline(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_audio_text.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_frame_pooled_audio_text
model:
  family: frame_pooled_audio_text
  text_encoder:
    type: static
    embedding_dim: 8
  audio:
    embedding_dim: 6
  frame_pooler:
    type: mean_std
    frame_dim: 4
  fusion:
    type: weighted_sum
    audio_dim: 6
    text_dim: 8
    final_dim: 5
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
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 1
    assert batch["waveforms"].shape == (2, 2, 1600)
    assert {tuple(sorted(group)) for group in batch["transcripts_grouped"]} == {
        ("transcript_spk1_0", "transcript_spk1_1"),
        ("transcript_spk2_0", "transcript_spk2_1"),
    }


def test_build_utterance_audio_text_data_pipeline(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_audio_text_flat.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_utterance_audio_text
model:
  family: utterance_audio_text_fusion
  text_encoder:
    type: static
    embedding_dim: 8
  audio_encoder:
    type: identity
  fusion:
    modalities: [audio, text]
    audio_dim: 1600
    text_dim: 8
    projection_dim: 8
    final_dim: 5
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 2
    assert batch["waveforms"].shape == (2, 1600)
    assert len(batch["transcripts"]) == 2


def test_build_utterance_audio_prosody_data_pipeline(tmp_path, monkeypatch) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path}\n")
    manifest = tmp_path / "train_audio_flat.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    monkeypatch.setattr(
        "multimodal_asv.data.audio.get_word_timestamps_for_audio",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "multimodal_asv.data.audio.extract_prosody_feature_tensor",
        lambda *args, **kwargs: torch.tensor([1.0, 2.0, 3.0]),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_utterance_audio_prosody
model:
  family: utterance_audio_text_fusion
  audio_encoder:
    type: identity
  fusion:
    modalities: [audio, pitch]
    audio_dim: 1600
    pitch_dim: 3
    projection_dim: 8
    final_dim: 5
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  num_workers: 0
  prosody_feature_set: pitch_rate_voicing
training:
  epochs: 1
  batch_size: 2
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 2
    assert batch["waveforms"].shape == (2, 1600)
    assert batch["pitch_features"].shape == (2, 3)


def test_build_hybrid_audio_text_data_pipeline(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2"):
        for utt in range(2):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path} transcript_{speaker}_{utt}\n")
    manifest = tmp_path / "train_audio_text_hybrid.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny_hybrid_audio_text
model:
  family: hybrid_audio_text
  audio_encoder:
    type: identity
  text_encoder:
    type: static
    embedding_dim: 8
  frame_dim: 4
  d_k: 8
  d_v: 8
  hybrid_dim: 8
  final_dim: 5
  text_dim: 8
  sluar_dim: 8
data:
  manifest: {manifest}
  sample_rate: 16000
  segment_seconds: 0.1
  num_workers: 0
training:
  epochs: 1
  batch_size: 2
""",
        encoding="utf-8",
    )

    pipeline = build_data_pipeline(load_config(config_path))
    batch = next(iter(pipeline.dataloader))

    assert len(pipeline.dataset) == 4
    assert len(pipeline.dataloader) == 2
    assert batch["waveforms"].shape == (2, 1600)
    assert len(batch["transcripts"]) == 2
