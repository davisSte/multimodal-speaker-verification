"""Text encoder builders for LUAR checkpoints and Hugging Face LUAR models."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from itertools import chain
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.utils.assets import resolve_asset_path


def _get_auto_model():
    from transformers import AutoModel

    return AutoModel


def _get_auto_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer


def _get_hf_hub_download():
    from huggingface_hub import hf_hub_download

    return hf_hub_download


def _get_vendor_luar():
    from multimodal_asv.vendor.luar_fallback import LUAR

    return LUAR


def _infer_luar_ft_metadata(checkpoint_path: str | Path) -> tuple[str, int]:
    """Recover metadata needed to rebuild a fine-tuned LUAR text encoder."""
    model_identifier = "sentence-transformers/paraphrase-distilroberta-base-v1"
    embedding_dim = 512
    try:
        with zipfile.ZipFile(checkpoint_path) as zf:
            data = zf.read("archive/data.pkl")
        candidates = re.findall(rb"sentence-transformers/[A-Za-z0-9._/-]+", data)
        if candidates:
            model_identifier = max(candidates, key=len).decode("utf-8").rstrip("r")
    except Exception as exc:
        logging.warning("Unable to infer LUAR-FT metadata from %s: %s", checkpoint_path, exc)
    return model_identifier, embedding_dim


class _LuarFtSelfAttention(nn.Module):
    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(k, q.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        return torch.matmul(F.softmax(scores, dim=-1), v)


class _LuarFtEpisodeEncoder(nn.Module):
    """Episode-level LUAR wrapper used when loading fine-tuned checkpoints."""

    def __init__(self, model_identifier: str, embedding_dim: int, *, cache_dir: str | None = None) -> None:
        super().__init__()
        self.transformer = _get_auto_model().from_pretrained(
            model_identifier,
            add_pooling_layer=False,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.hidden_size = int(self.transformer.config.hidden_size)
        self.attn_fn = _LuarFtSelfAttention()
        self.linear = nn.Linear(self.hidden_size, int(embedding_dim))

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = data
        batch_size, num_samples, episode_len, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_samples * episode_len, seq_len)
        flat_mask = attention_mask.reshape(batch_size * num_samples * episode_len, seq_len)
        outputs = self.transformer(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        comment_embeddings = self._mean_pool(outputs["last_hidden_state"], flat_mask)
        comment_embeddings = comment_embeddings.reshape(batch_size * num_samples, episode_len, self.hidden_size)
        episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = self.linear(episode_embeddings.max(dim=1).values)
        return episode_embeddings, comment_embeddings

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class LuarFTEncoder(nn.Module):
    """Fine-tuned LUAR checkpoint wrapper."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        token: str | None = None,
        freeze_model: bool = True,
        model_identifier: str | None = None,
        embedding_dim: int | None = None,
        checkpoint_url: str | None = None,
        download_root: str | Path | None = None,
        allow_download: bool = False,
    ) -> None:
        super().__init__()
        resolved_checkpoint = resolve_asset_path(
            checkpoint_path,
            allow_download=allow_download,
            download_url=checkpoint_url,
            download_root=download_root,
            default_filename="luar_checkpoint.ckpt",
            asset_name="LUAR-FT checkpoint",
        )
        if resolved_checkpoint is None or not resolved_checkpoint.exists():
            raise ValueError("LuarFTEncoder requires a valid local checkpoint_path or explicit checkpoint_url.")
        inferred_identifier, inferred_dim = _infer_luar_ft_metadata(resolved_checkpoint)
        self.model_identifier = model_identifier or inferred_identifier
        self.embedding_dim = int(embedding_dim or inferred_dim)
        cache_dir = str(Path(download_root).expanduser()) if download_root is not None else None
        self.model = _LuarFtEpisodeEncoder(self.model_identifier, self.embedding_dim, cache_dir=cache_dir)
        state = torch.load(resolved_checkpoint, map_location=torch.device("cpu"))
        if "state_dict" in state:
            state = state["state_dict"]
        drop_keys = {
            "transformer.pooler.dense.weight",
            "transformer.pooler.dense.bias",
            "transformer.embeddings.position_ids",
            "transformer.embeddings.token_type_ids",
        }
        state = {key: value for key, value in state.items() if key not in drop_keys}
        self.model.load_state_dict(state, strict=True)
        self.freeze_model = bool(freeze_model)
        if self.freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.tokenizer = _get_auto_tokenizer().from_pretrained(
            self.model_identifier,
            token=token,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=cache_dir,
        )

    def forward(
        self,
        texts: Sequence[str] | Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Encode either flat transcript lists or grouped speaker episodes."""
        if not texts:
            raise ValueError("LuarFTEncoder received no text.")
        if isinstance(texts[0], (list, tuple)):
            return self._encode_grouped(texts, max_length=max_length)  # type: ignore[arg-type]
        enc = self._tokenize(texts, max_length=max_length)  # type: ignore[arg-type]
        return self._run_model(enc["input_ids"].unsqueeze(1).unsqueeze(1), enc["attention_mask"].unsqueeze(1).unsqueeze(1))

    def _encode_grouped(
        self,
        grouped_texts: Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        lengths = [len(chunk) for chunk in grouped_texts]
        if any(length <= 0 for length in lengths):
            raise ValueError("Every speaker must contribute at least one transcript.")
        enc = self._tokenize(list(chain.from_iterable(grouped_texts)), max_length=max_length)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        batch = len(lengths)
        max_episode = max(lengths)
        seq_len = input_ids.size(-1)
        padded_ids = torch.zeros(batch, max_episode, seq_len, dtype=input_ids.dtype, device=input_ids.device)
        padded_mask = torch.zeros(batch, max_episode, seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        start = 0
        for index, length in enumerate(lengths):
            end = start + length
            padded_ids[index, :length] = input_ids[start:end]
            padded_mask[index, :length] = attention_mask[start:end]
            start = end
        return self._run_model(padded_ids.unsqueeze(1), padded_mask.unsqueeze(1))

    def _tokenize(self, texts: Sequence[str], *, max_length: int | None = None) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in enc.items()}

    def _run_model(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_model:
            with torch.no_grad():
                return self.model((input_ids, attention_mask))[0]
        return self.model((input_ids, attention_mask))[0]

    def get_input_embeddings(self) -> nn.Module:
        return self.model.transformer.get_input_embeddings()


class LuarHFEncoder(nn.Module):
    """Hugging Face LUAR wrapper."""

    def __init__(
        self,
        *,
        repo_id: str = "rrivera1849/LUAR-CRUD",
        token: str | None = None,
        freeze_model: bool = True,
        download_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        cache_dir = str(Path(download_root).expanduser()) if download_root is not None else None
        try:
            self.model = _get_auto_model().from_pretrained(
                repo_id,
                token=token,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            msg = str(exc)
            if "model type `luar`" not in msg and "model type 'luar'" not in msg and "KeyError: 'luar'" not in msg:
                raise
            cfg_path = _get_hf_hub_download()(repo_id=repo_id, filename="config.json", token=token, cache_dir=cache_dir)
            with open(cfg_path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            self.model = _get_vendor_luar().from_pretrained(
                repo_id,
                token=token,
                model_name_or_path=cfg.get(
                    "model_name_or_path",
                    "sentence-transformers/paraphrase-distilroberta-base-v1",
                ),
                embedding_dim=int(cfg.get("embedding_dim", 512)),
                epsilon=float(cfg.get("epsilon", 1e-9)),
                torch_dtype=cfg.get("torch_dtype", "bfloat16"),
                gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
                cache_dir=cache_dir,
            )
        self.freeze_model = bool(freeze_model)
        if self.freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.tokenizer = _get_auto_tokenizer().from_pretrained(
            repo_id,
            token=token,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=cache_dir,
        )

    def forward(
        self,
        texts: Sequence[str] | Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if not texts:
            raise ValueError("LuarHFEncoder received no text.")
        if isinstance(texts[0], (list, tuple)):
            return self._encode_grouped(texts, max_length=max_length)  # type: ignore[arg-type]
        enc = self._tokenize_flat(texts, max_length=max_length)  # type: ignore[arg-type]
        return self._run_model(enc)

    def _tokenize_flat(self, texts: Sequence[str], *, max_length: int | None = None) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc.pop("token_type_ids", None)
        device = next(self.model.parameters()).device
        return {
            "input_ids": enc["input_ids"].to(device).unsqueeze(1),
            "attention_mask": enc["attention_mask"].to(device).unsqueeze(1),
        }

    def _encode_grouped(
        self,
        grouped_texts: Sequence[Sequence[str]],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        lengths = [len(chunk) for chunk in grouped_texts]
        if any(length <= 0 for length in lengths):
            raise ValueError("Every speaker must contribute at least one transcript.")
        flat = list(chain.from_iterable(grouped_texts))
        enc = self.tokenizer(flat, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc.pop("token_type_ids", None)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        batch = len(lengths)
        max_episode = max(lengths)
        seq_len = input_ids.size(-1)
        device = next(self.model.parameters()).device
        padded_ids = torch.zeros(batch, max_episode, seq_len, dtype=input_ids.dtype, device=device)
        padded_mask = torch.zeros(batch, max_episode, seq_len, dtype=attention_mask.dtype, device=device)
        start = 0
        for index, length in enumerate(lengths):
            end = start + length
            padded_ids[index, :length] = input_ids[start:end].to(device)
            padded_mask[index, :length] = attention_mask[start:end].to(device)
            start = end
        return self._run_model({"input_ids": padded_ids, "attention_mask": padded_mask})

    def _run_model(self, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_model:
            with torch.no_grad():
                output = self.model(**encodings)
        else:
            output = self.model(**encodings)
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        raise RuntimeError(f"Unsupported LUAR output type: {type(output)}")

    def get_input_embeddings(self) -> nn.Module:
        if hasattr(self.model, "get_input_embeddings"):
            try:
                embeddings = self.model.get_input_embeddings()
            except NotImplementedError:
                embeddings = None
            if embeddings is not None:
                return embeddings
        transformer = getattr(self.model, "transformer", None)
        if transformer is not None and hasattr(transformer, "get_input_embeddings"):
            embeddings = transformer.get_input_embeddings()
            if embeddings is not None:
                return embeddings
        if transformer is not None and hasattr(transformer, "embeddings"):
            word_embeddings = getattr(transformer.embeddings, "word_embeddings", None)
            if word_embeddings is not None:
                return word_embeddings
        raise RuntimeError("LUAR model does not expose input embeddings.")


def build_luar_encoder(config: dict[str, object]) -> nn.Module:
    """Build either a fine-tuned LUAR encoder or a Hugging Face LUAR model."""
    checkpoint_path = config.get("checkpoint_path")
    checkpoint_url = config.get("checkpoint_url")
    if checkpoint_path or checkpoint_url:
        return LuarFTEncoder(
            checkpoint_path=str(checkpoint_path or "luar_checkpoint.ckpt"),
            token=_optional_str(config.get("token")),
            freeze_model=bool(config.get("freeze_model", True)),
            model_identifier=_optional_str(config.get("model_identifier")),
            embedding_dim=_optional_int(config.get("embedding_dim")),
            checkpoint_url=_optional_str(checkpoint_url),
            download_root=_optional_str(config.get("download_root")),
            allow_download=bool(config.get("allow_download", False)),
        )
    return LuarHFEncoder(
        repo_id=str(config.get("repo_id", "rrivera1849/LUAR-CRUD")),
        token=_optional_str(config.get("token")),
        freeze_model=bool(config.get("freeze_model", True)),
        download_root=_optional_str(config.get("download_root")),
    )


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = ["LuarFTEncoder", "LuarHFEncoder", "build_luar_encoder"]
