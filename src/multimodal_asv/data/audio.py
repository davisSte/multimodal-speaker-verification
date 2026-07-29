from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from multimodal_asv.utils.prosody import (
    extract_prosody_feature_tensor,
    get_word_timestamps_for_audio,
    load_whisper_transcript_index,
)


def read_audio_manifest(path: str | Path) -> list[tuple[str, Path]]:
    """Read a manifest with either `speaker_id audio_path` or `speaker_id|audio_path` rows."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: list[tuple[str, Path]] = []
    missing = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 2:
                    raise ValueError(f"Malformed manifest row {line_no} in {manifest_path}: {line!r}")
                speaker_id = parts[0]
                audio_path = Path(parts[1])
            else:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed manifest row {line_no} in {manifest_path}: {line!r}")
                speaker_id = parts[0]
                audio_path = Path(parts[1])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            if not audio_path.exists():
                missing += 1
                logging.warning("Skipping missing audio path from %s: %s", manifest_path, audio_path)
                continue
            samples.append((speaker_id, audio_path))

    if not samples:
        raise ValueError(f"No usable entries found in {manifest_path}")
    if missing:
        logging.info("Skipped %d missing audio entries from %s", missing, manifest_path)
    return samples


def read_audio_text_manifest(path: str | Path) -> list[tuple[str, Path, str]]:
    """Read an audio-text manifest in whitespace or pipe-delimited repo formats."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: list[tuple[str, Path, str]] = []
    missing = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 4:
                    _utt_id, audio_path_raw, transcript, speaker_id = parts[:4]
                elif len(parts) >= 3:
                    speaker_id, audio_path_raw, transcript = parts[:3]
                else:
                    raise ValueError(
                        f"Malformed audio-text manifest row {line_no} in {manifest_path}: {line!r}"
                    )
                audio_path = Path(audio_path_raw)
            else:
                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    raise ValueError(
                        f"Malformed audio-text manifest row {line_no} in {manifest_path}: {line!r}"
                    )
                speaker_id = parts[0]
                audio_path = Path(parts[1])
                transcript = parts[2]
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            if not audio_path.exists():
                missing += 1
                logging.warning("Skipping missing audio path from %s: %s", manifest_path, audio_path)
                continue
            samples.append((speaker_id, audio_path, transcript))

    if not samples:
        raise ValueError(f"No usable entries found in {manifest_path}")
    if missing:
        logging.info("Skipped %d missing audio-text entries from %s", missing, manifest_path)
    return samples


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio as a waveform tensor shaped `(channels, samples)`."""

    return torchaudio.load(str(path))


class AudioUtteranceDataset(Dataset):
    """Load cropped audio utterances with speaker labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 4.0,
        random_segment: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(self.sample_rate * self.segment_seconds))
        if self.segment_samples <= 0:
            raise ValueError("segment_seconds must be positive.")
        self.random_segment = bool(random_segment)
        self.rng = random.Random(seed)

        self.samples = read_audio_manifest(self.manifest_path)
        speakers = sorted({speaker for speaker, _path in self.samples})
        if not speakers:
            raise ValueError("No speakers found in manifest.")
        self.speaker_to_idx = {speaker: index for index, speaker in enumerate(speakers)}
        self.idx_to_speaker = {index: speaker for speaker, index in self.speaker_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        speaker_id, path = self.samples[index]
        waveform, sample_rate = load_audio(path)
        waveform = self._to_mono(waveform)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sample_rate,
            ).squeeze(0)
        waveform, valid_length = self._crop_or_pad(waveform)
        return {
            "waveform": waveform,
            "length": torch.tensor(valid_length, dtype=torch.long),
            "label": torch.tensor(self.speaker_to_idx[speaker_id], dtype=torch.long),
            "speaker_id": speaker_id,
            "path": str(path),
        }

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.numel() == 0:
            return torch.zeros(1)
        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_samples = waveform.size(-1)
        target = self.segment_samples
        valid_length = min(num_samples, target)
        if num_samples > target:
            if self.random_segment:
                start = self.rng.randint(0, num_samples - target)
            else:
                start = (num_samples - target) // 2
            waveform = waveform[start : start + target]
        elif num_samples < target:
            waveform = F.pad(waveform, (0, target - num_samples))
        return waveform.contiguous(), valid_length


class AudioUtteranceProsodyDataset(AudioUtteranceDataset):
    """Grouped audio dataset that also attaches per-utterance prosody features."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 4.0,
        random_segment: bool = True,
        seed: int | None = None,
        prosody_feature_set: str = "pitch_rate_voicing",
        prosody_transcript_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            manifest_path,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            random_segment=random_segment,
            seed=seed,
        )
        self.prosody_feature_set = str(prosody_feature_set)
        self.transcript_index = (
            load_whisper_transcript_index(prosody_transcript_path)
            if prosody_transcript_path is not None
            else None
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = super().__getitem__(index)
        word_timestamps = get_word_timestamps_for_audio(
            item["path"],
            transcript_index=self.transcript_index,
        )
        item["pitch_features"] = extract_prosody_feature_tensor(
            _as_tensor(item["waveform"]),
            self.sample_rate,
            feature_set=self.prosody_feature_set,
            word_timestamps=word_timestamps,
        )
        return item


class AudioTextUtteranceDataset(Dataset):
    """Load cropped audio utterances plus aligned transcripts."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 4.0,
        random_segment: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(self.sample_rate * self.segment_seconds))
        if self.segment_samples <= 0:
            raise ValueError("segment_seconds must be positive.")
        self.random_segment = bool(random_segment)
        self.rng = random.Random(seed)

        self.samples = read_audio_text_manifest(self.manifest_path)
        speakers = sorted({speaker for speaker, _path, _transcript in self.samples})
        if not speakers:
            raise ValueError("No speakers found in manifest.")
        self.speaker_to_idx = {speaker: index for index, speaker in enumerate(speakers)}
        self.idx_to_speaker = {index: speaker for speaker, index in self.speaker_to_idx.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        speaker_id, path, transcript = self.samples[index]
        waveform, sample_rate = load_audio(path)
        waveform = AudioUtteranceDataset._to_mono(waveform)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sample_rate,
            ).squeeze(0)
        waveform, valid_length = self._crop_or_pad(waveform)
        return {
            "waveform": waveform,
            "length": torch.tensor(valid_length, dtype=torch.long),
            "label": torch.tensor(self.speaker_to_idx[speaker_id], dtype=torch.long),
            "speaker_id": speaker_id,
            "path": str(path),
            "transcript": transcript,
        }

    def _crop_or_pad(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_samples = waveform.size(-1)
        target = self.segment_samples
        valid_length = min(num_samples, target)
        if num_samples > target:
            if self.random_segment:
                start = self.rng.randint(0, num_samples - target)
            else:
                start = (num_samples - target) // 2
            waveform = waveform[start : start + target]
        elif num_samples < target:
            waveform = F.pad(waveform, (0, target - num_samples))
        return waveform.contiguous(), valid_length


def pad_audio_batch(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    lengths = torch.stack([_as_long_tensor(item["length"]) for item in batch])
    max_len = int(lengths.max().item())
    waveforms = []
    for item in batch:
        waveform = _as_tensor(item["waveform"])
        if waveform.numel() < max_len:
            waveform = F.pad(waveform, (0, max_len - waveform.numel()))
        waveforms.append(waveform)
    return {
        "waveforms": torch.stack(waveforms, dim=0),
        "lengths": lengths,
        "labels": torch.stack([_as_long_tensor(item["label"]) for item in batch]),
        "speaker_ids": [str(item["speaker_id"]) for item in batch],
        "paths": [str(item["path"]) for item in batch],
    }


def pad_audio_text_batch(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    packed = pad_audio_batch(batch)
    return {
        **packed,
        "transcripts": [str(item["transcript"]) for item in batch],
    }


def pad_audio_prosody_batch(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    packed = pad_audio_batch(batch)
    return {
        **packed,
        "pitch_features": torch.stack([_as_tensor(item["pitch_features"]) for item in batch], dim=0),
    }


def pad_grouped_audio_batch(
    batch: Sequence[dict[str, torch.Tensor | str]],
    *,
    speakers_per_batch: int,
    utterances_per_speaker: int,
) -> dict[str, torch.Tensor | list[str]]:
    packed = pad_audio_batch(batch)
    expected = speakers_per_batch * utterances_per_speaker
    if packed["waveforms"].size(0) != expected:
        raise ValueError(f"Expected {expected} utterances, got {packed['waveforms'].size(0)}.")
    return {
        "waveforms": packed["waveforms"].view(speakers_per_batch, utterances_per_speaker, -1),
        "lengths": packed["lengths"].view(speakers_per_batch, utterances_per_speaker),
        "labels": packed["labels"].view(speakers_per_batch, utterances_per_speaker),
        "speaker_ids": packed["speaker_ids"],
        "paths": packed["paths"],
        **(
            {
                "pitch_features_grouped": torch.stack(
                    [_as_tensor(item["pitch_features"]) for item in batch],
                    dim=0,
                ).view(speakers_per_batch, utterances_per_speaker, -1)
            }
            if "pitch_features" in batch[0]
            else {}
        ),
    }


def pad_grouped_audio_text_batch(
    batch: Sequence[dict[str, torch.Tensor | str]],
    *,
    speakers_per_batch: int,
    utterances_per_speaker: int,
) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
    packed = pad_audio_batch(batch)
    expected = speakers_per_batch * utterances_per_speaker
    if packed["waveforms"].size(0) != expected:
        raise ValueError(f"Expected {expected} utterances, got {packed['waveforms'].size(0)}.")
    transcripts = [str(item["transcript"]) for item in batch]
    transcripts_grouped = [
        transcripts[index * utterances_per_speaker : (index + 1) * utterances_per_speaker]
        for index in range(speakers_per_batch)
    ]
    return {
        "waveforms": packed["waveforms"].view(speakers_per_batch, utterances_per_speaker, -1),
        "lengths": packed["lengths"].view(speakers_per_batch, utterances_per_speaker),
        "labels": packed["labels"].view(speakers_per_batch, utterances_per_speaker),
        "speaker_ids": packed["speaker_ids"],
        "paths": packed["paths"],
        "transcripts_grouped": transcripts_grouped,
    }


class GroupedSpeakerBatchSampler(Sampler[list[int]]):
    """Sample batches with several speakers and several utterances per speaker."""

    def __init__(
        self,
        dataset: AudioUtteranceDataset,
        *,
        speakers_per_batch: int,
        utterances_per_speaker: int,
        seed: int | None = None,
    ) -> None:
        if speakers_per_batch <= 0 or utterances_per_speaker <= 0:
            raise ValueError("speakers_per_batch and utterances_per_speaker must be positive.")
        self.dataset = dataset
        self.speakers_per_batch = int(speakers_per_batch)
        self.utterances_per_speaker = int(utterances_per_speaker)
        self.seed = int(seed or 0)
        self.epoch = 0
        self.speaker_to_indices = self._build_index()
        self.speaker_to_indices = {
            speaker: indices
            for speaker, indices in self.speaker_to_indices.items()
            if len(indices) >= self.utterances_per_speaker
        }
        if len(self.speaker_to_indices) < self.speakers_per_batch:
            raise ValueError(
                "Not enough speakers have at least "
                f"{self.utterances_per_speaker} utterances to form a batch."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self.speaker_to_indices.values())
        return total // (self.speakers_per_batch * self.utterances_per_speaker)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        speaker_ids = list(self.speaker_to_indices)
        for _ in range(len(self)):
            rng.shuffle(speaker_ids)
            selected = speaker_ids[: self.speakers_per_batch]
            batch: list[int] = []
            for speaker in selected:
                batch.extend(rng.sample(self.speaker_to_indices[speaker], self.utterances_per_speaker))
            yield batch

    def _build_index(self) -> dict[int, list[int]]:
        mapping: dict[int, list[int]] = {}
        for index, sample in enumerate(self.dataset.samples):
            speaker_id = sample[0]
            speaker_index = self.dataset.speaker_to_idx[speaker_id]
            mapping.setdefault(speaker_index, []).append(index)
        return mapping


def build_grouped_audio_dataloader(
    *,
    manifest_path: str | Path,
    speakers_per_batch: int,
    utterances_per_speaker: int,
    sample_rate: int = 16_000,
    segment_seconds: float = 4.0,
    random_segment: bool = True,
    num_workers: int = 4,
    seed: int | None = None,
) -> tuple[AudioUtteranceDataset, DataLoader]:
    dataset = AudioUtteranceDataset(
        manifest_path,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        random_segment=random_segment,
        seed=seed,
    )
    batch_sampler = GroupedSpeakerBatchSampler(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_grouped_audio_batch(
            batch,
            speakers_per_batch=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        ),
    )
    return dataset, dataloader


def build_grouped_audio_prosody_dataloader(
    *,
    manifest_path: str | Path,
    speakers_per_batch: int,
    utterances_per_speaker: int,
    sample_rate: int = 16_000,
    segment_seconds: float = 4.0,
    random_segment: bool = True,
    num_workers: int = 4,
    seed: int | None = None,
    prosody_feature_set: str = "pitch_rate_voicing",
    prosody_transcript_path: str | Path | None = None,
) -> tuple[AudioUtteranceProsodyDataset, DataLoader]:
    dataset = AudioUtteranceProsodyDataset(
        manifest_path,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        random_segment=random_segment,
        seed=seed,
        prosody_feature_set=prosody_feature_set,
        prosody_transcript_path=prosody_transcript_path,
    )
    batch_sampler = GroupedSpeakerBatchSampler(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_grouped_audio_batch(
            batch,
            speakers_per_batch=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        ),
    )
    return dataset, dataloader


def build_flat_audio_text_dataloader(
    *,
    manifest_path: str | Path,
    batch_size: int,
    sample_rate: int = 16_000,
    segment_seconds: float = 4.0,
    random_segment: bool = True,
    num_workers: int = 4,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[AudioTextUtteranceDataset, DataLoader]:
    dataset = AudioTextUtteranceDataset(
        manifest_path,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        random_segment=random_segment,
        seed=seed,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_audio_text_batch,
    )
    return dataset, dataloader


def build_flat_audio_prosody_dataloader(
    *,
    manifest_path: str | Path,
    batch_size: int,
    sample_rate: int = 16_000,
    segment_seconds: float = 4.0,
    random_segment: bool = True,
    num_workers: int = 4,
    seed: int | None = None,
    shuffle: bool = True,
    prosody_feature_set: str = "pitch_rate_voicing",
    prosody_transcript_path: str | Path | None = None,
) -> tuple[AudioUtteranceProsodyDataset, DataLoader]:
    dataset = AudioUtteranceProsodyDataset(
        manifest_path,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        random_segment=random_segment,
        seed=seed,
        prosody_feature_set=prosody_feature_set,
        prosody_transcript_path=prosody_transcript_path,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_audio_prosody_batch,
    )
    return dataset, dataloader


def build_grouped_audio_text_dataloader(
    *,
    manifest_path: str | Path,
    speakers_per_batch: int,
    utterances_per_speaker: int,
    sample_rate: int = 16_000,
    segment_seconds: float = 4.0,
    random_segment: bool = True,
    num_workers: int = 4,
    seed: int | None = None,
) -> tuple[AudioTextUtteranceDataset, DataLoader]:
    dataset = AudioTextUtteranceDataset(
        manifest_path,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        random_segment=random_segment,
        seed=seed,
    )
    batch_sampler = GroupedSpeakerBatchSampler(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: pad_grouped_audio_text_batch(
            batch,
            speakers_per_batch=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        ),
    )
    return dataset, dataloader


def _as_tensor(value: torch.Tensor | str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor, got {type(value).__name__}.")
    return value


def _as_long_tensor(value: torch.Tensor | str) -> torch.Tensor:
    tensor = _as_tensor(value)
    return tensor.long()


__all__ = [
    "AudioUtteranceDataset",
    "AudioTextUtteranceDataset",
    "AudioUtteranceProsodyDataset",
    "GroupedSpeakerBatchSampler",
    "build_flat_audio_prosody_dataloader",
    "build_flat_audio_text_dataloader",
    "build_grouped_audio_dataloader",
    "build_grouped_audio_prosody_dataloader",
    "build_grouped_audio_text_dataloader",
    "load_audio",
    "pad_audio_batch",
    "pad_audio_prosody_batch",
    "pad_audio_text_batch",
    "pad_grouped_audio_batch",
    "pad_grouped_audio_text_batch",
    "read_audio_manifest",
    "read_audio_text_manifest",
]
