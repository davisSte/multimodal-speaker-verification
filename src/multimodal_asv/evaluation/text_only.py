from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from multimodal_asv.encoders.text import build_luar_encoder
from .common import extract_texts_from_side


def build_text_only_luar_encoder(
    *,
    repo_id: str,
    checkpoint_path: Path | None,
    hf_token: str | None,
    device: str,
) -> torch.nn.Module:
    encoder = build_luar_encoder(
        {
            "repo_id": repo_id,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "token": hf_token,
            "freeze_model": True,
        }
    ).to(device)
    encoder.eval()
    return encoder


def encode_text_episode(
    encoder: torch.nn.Module,
    texts: Sequence[str],
    cache: Dict[Tuple[str, ...], torch.Tensor],
) -> torch.Tensor:
    key = tuple(texts)
    cached = cache.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        emb = encoder([list(texts)])
    if not isinstance(emb, torch.Tensor):
        raise RuntimeError(f"Encoder returned non-tensor output: {type(emb)}")
    if emb.ndim == 2:
        if emb.size(0) != 1:
            raise RuntimeError(f"Expected one episode embedding, got shape={tuple(emb.shape)}")
        emb = emb[0]
    elif emb.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape={tuple(emb.shape)}")
    emb = F.normalize(emb.to(torch.float32).detach().cpu(), dim=0)
    cache[key] = emb
    return emb


def extract_text_side(side: dict[str, Any]) -> list[str]:
    return extract_texts_from_side(side)
