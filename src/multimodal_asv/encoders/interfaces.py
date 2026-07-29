from __future__ import annotations

from typing import Protocol, Sequence

import torch


class AudioEncoder(Protocol):
    """Callable interface for modules that map waveforms to utterance embeddings."""

    def __call__(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class TextEncoder(Protocol):
    """Callable interface for modules that map transcripts to utterance embeddings."""

    def __call__(
        self,
        transcripts: Sequence[str] | Sequence[Sequence[str]],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]: ...


__all__ = ["AudioEncoder", "TextEncoder"]
