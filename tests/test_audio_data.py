from __future__ import annotations

import torch
import torchaudio

from multimodal_asv.data import AudioUtteranceDataset, build_grouped_audio_dataloader
from multimodal_asv.data.audio import read_audio_text_manifest


def _write_wav(path, samples: int, sample_rate: int = 16000) -> None:
    waveform = torch.linspace(-0.5, 0.5, steps=samples).unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)


def test_audio_utterance_dataset_loads_and_crops(tmp_path) -> None:
    wav_path = tmp_path / "spk1_utt1.wav"
    _write_wav(wav_path, samples=3200)
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"spk1 {wav_path}\n", encoding="utf-8")

    dataset = AudioUtteranceDataset(
        manifest,
        sample_rate=16000,
        segment_seconds=0.1,
        random_segment=False,
    )
    item = dataset[0]

    assert item["waveform"].shape == (1600,)
    assert int(item["length"].item()) == 1600
    assert int(item["label"].item()) == 0
    assert item["speaker_id"] == "spk1"


def test_grouped_audio_dataloader_shapes(tmp_path) -> None:
    rows = []
    for speaker in ("spk1", "spk2", "spk3"):
        for utt in range(3):
            wav_path = tmp_path / f"{speaker}_{utt}.wav"
            _write_wav(wav_path, samples=1600)
            rows.append(f"{speaker} {wav_path}\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("".join(rows), encoding="utf-8")

    _dataset, loader = build_grouped_audio_dataloader(
        manifest_path=manifest,
        speakers_per_batch=2,
        utterances_per_speaker=2,
        sample_rate=16000,
        segment_seconds=0.1,
        random_segment=False,
        num_workers=0,
        seed=1234,
    )
    batch = next(iter(loader))

    assert batch["waveforms"].shape == (2, 2, 1600)
    assert batch["lengths"].shape == (2, 2)
    assert batch["labels"].shape == (2, 2)


def test_read_audio_text_manifest_supports_pipe_delimited_hybrid_rows(tmp_path) -> None:
    wav_path = tmp_path / "spk1_utt1.wav"
    _write_wav(wav_path, samples=1600)
    manifest = tmp_path / "train_audio_text_pipe.txt"
    manifest.write_text(
        f"utt_spk1_001|{wav_path}|hello there|spk1\n",
        encoding="utf-8",
    )

    rows = read_audio_text_manifest(manifest)

    assert rows == [("spk1", wav_path, "hello there")]
