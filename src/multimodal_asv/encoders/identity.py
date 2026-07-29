from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class IdentityAudioEncoder(nn.Module):
    """Test helper that treats input waveforms/features as already-encoded embeddings."""

    def forward(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        return waveforms


class StaticTextEncoder(nn.Module):
    """Small deterministic text encoder for tests and examples."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, transcripts: Sequence[str] | Sequence[Sequence[str]]) -> torch.Tensor:
        rows = []
        for transcript in transcripts:
            if isinstance(transcript, (list, tuple)):
                value = float(sum(len(item.strip()) for item in transcript))
            else:
                value = float(len(transcript.strip()))
            rows.append(torch.full((self.embedding_dim,), value))
        return torch.stack(rows, dim=0)


__all__ = ["IdentityAudioEncoder", "StaticTextEncoder"]
