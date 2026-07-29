from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import pdb
import time

import torch
from torch import nn

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.training.losses import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import resolve_output_dir


class TrainableModel(Protocol):
    def train(self, mode: bool = True): ...


@dataclass
class Trainer:
    """Small training facade that will host shared training behavior."""

    config: ExperimentConfig

    def fit(self, model: TrainableModel, train_loader) -> list[dict[str, float]]:
        model.train(True)
        device = self._resolve_device()
        model = model.to(device)
        speaker_loss = self._ensure_speaker_loss(model, train_loader)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters found.")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        output_dir = self._prepare_output_dir()
        log_interval = int(self.config.training.options.get("log_interval", 10))
        debug_pdb_first_batch = bool(self.config.training.options.get("debug_pdb_first_batch", False))
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.training.epochs + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)

            total_loss = 0.0
            total_acc = 0.0
            total_batches = 0
            start_time = time.time()

            for step, batch in enumerate(train_loader, start=1):
                if debug_pdb_first_batch and epoch == 1 and step == 1:
                    pdb.set_trace()
                optimizer.zero_grad(set_to_none=True)
                if "transcripts_grouped" in batch:
                    embeddings, labels = self._forward_grouped_audio_text_batch(model, batch, device)
                elif "pitch_features_grouped" in batch:
                    embeddings, labels = self._forward_grouped_audio_prosody_batch(model, batch, device)
                elif "pitch_features" in batch:
                    embeddings, labels = self._forward_flat_audio_prosody_batch(model, batch, device)
                elif "transcripts" in batch:
                    embeddings, labels = self._forward_flat_audio_text_batch(model, batch, device)
                else:
                    embeddings, labels = self._forward_grouped_audio_batch(model, batch, device)
                loss, prec1 = speaker_loss(embeddings, labels)
                loss.backward()
                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.training.grad_clip)
                optimizer.step()

                total_loss += float(loss.item())
                total_acc += float(prec1.item())
                total_batches += 1

                if log_interval > 0 and (step % log_interval == 0 or step == len(train_loader)):
                    print(
                        f"epoch={epoch} step={step}/{len(train_loader)} "
                        f"loss={loss.item():.4f} acc1={prec1.item():.2f}",
                        flush=True,
                    )

            metrics = {
                "epoch": float(epoch),
                "loss": total_loss / max(1, total_batches),
                "acc1": total_acc / max(1, total_batches),
                "elapsed_seconds": time.time() - start_time,
            }
            history.append(metrics)
            print(
                f"epoch={epoch} complete loss={metrics['loss']:.4f} "
                f"acc1={metrics['acc1']:.2f} elapsed={metrics['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if output_dir is not None:
                self._save_checkpoint(output_dir, model, optimizer, epoch, metrics)

        return history

    def _resolve_device(self) -> torch.device:
        requested = str(self.config.training.options.get("device", "cpu")).strip().lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    def _prepare_output_dir(self) -> Path | None:
        output_dir = resolve_output_dir(self.config.output_dir, experiment_name=self.config.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _ensure_speaker_loss(self, model: nn.Module, train_loader) -> AdditiveAngularMarginSoftmax:
        speaker_loss = getattr(model, "speaker_loss", None)
        if speaker_loss is not None:
            return speaker_loss

        dataset = getattr(train_loader, "dataset", None)
        num_classes = None
        if dataset is not None and hasattr(dataset, "speaker_to_idx"):
            num_classes = len(dataset.speaker_to_idx)
        embedding_dim = int(getattr(model, "embedding_dim", 192))
        if num_classes is None:
            raise ValueError("Could not infer speaker count to attach speaker_loss.")
        speaker_loss = AdditiveAngularMarginSoftmax(
            num_classes=num_classes,
            margin=0.2,
            scale=30.0,
            embedding_dim=embedding_dim,
        )
        model.speaker_loss = speaker_loss.to(next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
        return model.speaker_loss

    def _forward_grouped_audio_batch(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        speakers_per_batch, utterances_per_speaker, max_len = waveforms.shape
        padding_mask = self._make_padding_mask(lengths.view(-1), max_len, device).view(
            speakers_per_batch,
            utterances_per_speaker,
            max_len,
        )
        embeddings = model(
            waveforms=waveforms,
            padding_mask=padding_mask,
            num_speakers=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        )
        speaker_labels = self._speaker_labels(labels, speakers_per_batch, utterances_per_speaker)
        return embeddings, speaker_labels

    def _forward_grouped_audio_text_batch(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        speakers_per_batch, utterances_per_speaker, max_len = waveforms.shape
        padding_mask = self._make_padding_mask(lengths.view(-1), max_len, device).view(
            speakers_per_batch,
            utterances_per_speaker,
            max_len,
        )
        embeddings = model(
            waveforms=waveforms,
            transcripts_grouped=batch["transcripts_grouped"],
            padding_mask=padding_mask,
            num_speakers=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        )
        speaker_labels = self._speaker_labels(labels, speakers_per_batch, utterances_per_speaker)
        return embeddings, speaker_labels

    def _forward_grouped_audio_prosody_batch(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        pitch_features_grouped = batch["pitch_features_grouped"].to(device)
        speakers_per_batch, utterances_per_speaker, max_len = waveforms.shape
        padding_mask = self._make_padding_mask(lengths.view(-1), max_len, device).view(
            speakers_per_batch,
            utterances_per_speaker,
            max_len,
        )
        embeddings = model(
            waveforms=waveforms,
            padding_mask=padding_mask,
            pitch_features_grouped=pitch_features_grouped,
            num_speakers=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
        )
        speaker_labels = self._speaker_labels(labels, speakers_per_batch, utterances_per_speaker)
        return embeddings, speaker_labels

    def _forward_flat_audio_text_batch(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = self._make_padding_mask(lengths, waveforms.size(1), device)
        embeddings = model(
            waveforms=waveforms,
            transcripts=batch["transcripts"],
            padding_mask=padding_mask,
        )
        return embeddings, labels

    def _forward_flat_audio_prosody_batch(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        pitch_features = batch["pitch_features"].to(device)
        padding_mask = self._make_padding_mask(lengths, waveforms.size(1), device)
        embeddings = model(
            waveforms=waveforms,
            padding_mask=padding_mask,
            pitch_features=pitch_features,
        )
        return embeddings, labels

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        rng = torch.arange(max_len, device=device).unsqueeze(0)
        return rng >= lengths.unsqueeze(1)

    @staticmethod
    def _speaker_labels(labels: torch.Tensor, num_speakers: int, utterances_per_speaker: int) -> torch.Tensor:
        reshaped = labels.view(num_speakers, utterances_per_speaker)
        return reshaped[:, 0].contiguous()

    def _save_checkpoint(
        self,
        output_dir: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config_name": self.config.name,
        }
        path = output_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
