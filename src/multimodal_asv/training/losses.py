from __future__ import annotations

import math
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)) -> list[torch.Tensor]:
    """Compute top-k classification accuracy percentages."""

    if target.numel() == 0:
        return [torch.tensor(0.0, device=output.device)]
    topk = tuple(topk)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


class AdditiveAngularMarginSoftmax(nn.Module):
    """AAM-Softmax speaker classification loss."""

    def __init__(self, num_classes: int, margin: float, scale: float, embedding_dim: int = 192) -> None:
        super().__init__()
        self.margin = float(margin)
        self.scale = float(scale)
        self.embedding_dim = int(embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim), requires_grad=True)
        self.cross_entropy = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_adjustment = math.sin(math.pi - self.margin) * self.margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.threshold) > 0, phi, cosine - self.margin_adjustment)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale

        self._emit_debug_trace_once(embeddings, labels, cosine, logits)

        loss = self.cross_entropy(logits, labels)
        prec1 = accuracy(logits.detach(), labels.detach(), topk=(1,))[0]
        if return_logits:
            return loss, prec1, logits
        return loss, prec1

    def _emit_debug_trace_once(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        cosine: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if os.getenv("DEBUG_AAM_TRACE", "0") != "1" or hasattr(self, "_aam_trace_emitted"):
            return
        self._aam_trace_emitted = True

        def _stats(name: str, tensor: torch.Tensor) -> None:
            finite = bool(torch.isfinite(tensor).all().item())
            detached = tensor.detach()
            print(
                f"[AAM_TRACE] {name}: shape={tuple(tensor.shape)} finite={finite} "
                f"min={float(detached.min().item()):.6g} "
                f"max={float(detached.max().item()):.6g} "
                f"mean={float(detached.float().mean().item()):.6g}",
                flush=True,
            )

        _stats("embeddings", embeddings)
        _stats("weight", self.weight)
        _stats("cosine", cosine)
        _stats("logits", logits)
        print(
            f"[AAM_TRACE] labels: min={int(labels.min().item())} "
            f"max={int(labels.max().item())} "
            f"n_unique={int(labels.unique().numel())} n_class={self.weight.shape[0]}",
            flush=True,
        )
