"""Hybrid acoustic-textual models with token-to-frame cross-attention alignment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_asv.config import ExperimentConfig
from multimodal_asv.encoders import build_audio_encoder, build_text_encoder
from multimodal_asv.training import AdditiveAngularMarginSoftmax
from multimodal_asv.utils.assets import with_default_download_root


class CrossAttentionAligner(nn.Module):
    """Text-token queries aligned against acoustic frame features.

    Input shapes:
    - text_embeddings: (batch, tokens, text_dim)
    - frame_map: (batch, frame_dim, frames)

    Output shape:
    - attended token-conditioned acoustic features: (batch, tokens, d_v)
    """

    def __init__(
        self,
        *,
        text_dim: int,
        frame_dim: int,
        d_k: int = 256,
        d_v: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_k % num_heads != 0:
            raise ValueError("d_k must be divisible by num_heads.")
        if d_v % num_heads != 0:
            raise ValueError("d_v must be divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.head_dk = self.d_k // self.num_heads
        self.head_dv = self.d_v // self.num_heads

        self.query_proj = nn.Linear(text_dim, self.d_k)
        self.key_proj = nn.Linear(frame_dim, self.d_k)
        self.value_proj = nn.Linear(frame_dim, self.d_v)
        self.output_proj = nn.Linear(self.d_v, self.d_v)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        frame_map: torch.Tensor,
        *,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = text_embeddings.shape
        num_frames = frame_map.size(-1)
        # The audio encoder returns channel-first frame features; attention is
        # easier to write in token/frame-major form.
        frames = frame_map.transpose(1, 2)

        queries = self.query_proj(text_embeddings)
        keys = self.key_proj(frames)
        values = self.value_proj(frames)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dk).transpose(1, 2)
        keys = keys.view(batch_size, num_frames, self.num_heads, self.head_dk).transpose(1, 2)
        values = values.view(batch_size, num_frames, self.num_heads, self.head_dv).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dk**0.5)
        if text_padding_mask is not None:
            # We mask special/padded text positions so they do not aggregate
            # over the acoustic frames.
            blocked = text_padding_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, num_frames)
            all_blocked = blocked.all(dim=-1, keepdim=True)
            scores = scores.masked_fill(blocked, -1e9)
            scores = scores.masked_fill(all_blocked, 0.0)
        else:
            all_blocked = None

        weights = F.softmax(scores, dim=-1)
        if all_blocked is not None:
            weights = weights.masked_fill(all_blocked, 0.0)
        attended = torch.matmul(weights, values)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_v)
        return self.output_proj(attended)


class HybridAudioTextModel(nn.Module):
    """
    Plain hybrid audio-text model from the LUAR cross-attention experiments.

    The model uses LUAR token embeddings as text queries over WavLM-ECAPA frame
    features, pools the token sequence, then fuses that hybrid branch with a
    LUAR utterance embedding.

    High-level flow:
    1. Audio encoder produces frame-level acoustic features.
    2. LUAR tokenizer + embedding table produce lexical token embeddings.
    3. Cross-attention aligns each token against all acoustic frames.
    4. Lexical token features and aligned acoustic token features are fused.
    5. Optional token self-attention refines the token sequence.
    6. The token sequence is pooled into one hybrid utterance embedding.
    7. Optional utterance-level LUAR embedding is concatenated in as a second
       text stream before the final speaker projection.
    """

    def __init__(
        self,
        *,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        frame_dim: int = 1536,
        d_k: int = 256,
        d_v: int = 256,
        fusion_mode: str = "concat",
        cross_attn_heads: int = 4,
        self_attn_heads: int = 4,
        self_attn_layers: int = 1,
        use_token_self_attention: bool = True,
        use_luar_stream: bool = True,
        use_stat_pooling: bool = False,
        hybrid_dim: int = 256,
        final_dim: int = 256,
        text_dim: int | None = None,
        sluar_dim: int = 512,
        normalize: bool = True,
        debug_pdb_stages: Sequence[str] | None = None,
        num_classes: int | None = None,
        speaker_loss_margin: float = 0.2,
        speaker_loss_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.use_token_self_attention = bool(use_token_self_attention)
        self.use_luar_stream = bool(use_luar_stream)
        self.use_stat_pooling = bool(use_stat_pooling)
        self.hybrid_dim = int(hybrid_dim)
        self.embedding_dim = int(final_dim)
        self.normalize = bool(normalize)
        self.debug_pdb_stages = {str(stage) for stage in (debug_pdb_stages or [])}
        self._token_embed_table = self._resolve_token_embedding_table()
        self._special_token_ids = self._resolve_special_token_ids()
        self.text_dim = int(text_dim if text_dim is not None else self._token_embedding_dim())

        self.cross_attention = CrossAttentionAligner(
            text_dim=self.text_dim,
            frame_dim=int(frame_dim),
            d_k=int(d_k),
            d_v=int(d_v),
            num_heads=int(cross_attn_heads),
        )

        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode == "concat":
            # Preserve both lexical identity and attended acoustic evidence by
            # concatenating them token-wise before projection.
            self.fusion_proj = nn.Linear(self.text_dim + int(d_v), self.hybrid_dim)
            self.text_to_dv = None
            self.fusion_layer_norm = None
        elif self.fusion_mode == "residual_add":
            # Force text and aligned audio into the same space and combine them
            # additively before projecting to the hybrid token space.
            self.text_to_dv = nn.Linear(self.text_dim, int(d_v))
            self.fusion_layer_norm = nn.LayerNorm(int(d_v))
            self.fusion_proj = nn.Linear(int(d_v), self.hybrid_dim)
        else:
            raise ValueError(f"Unsupported hybrid fusion mode: {fusion_mode!r}")

        if self.use_token_self_attention:
            if self.hybrid_dim % int(self_attn_heads) != 0:
                raise ValueError("hybrid_dim must be divisible by self_attn_heads when token self-attention is enabled.")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hybrid_dim,
                nhead=int(self_attn_heads),
                dim_feedforward=self.hybrid_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.token_self_attention = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(self_attn_layers),
            )
        else:
            self.token_self_attention = nn.Identity()

        self.sluar_proj = nn.Linear(int(sluar_dim), self.hybrid_dim) if self.use_luar_stream else None
        pooled_dim = self.hybrid_dim * (2 if self.use_stat_pooling else 1)
        final_in_dim = pooled_dim + (self.hybrid_dim if self.use_luar_stream else 0)
        self.final_proj = nn.Linear(final_in_dim, self.embedding_dim)
        self.speaker_loss = (
            AdditiveAngularMarginSoftmax(
                num_classes=num_classes,
                margin=speaker_loss_margin,
                scale=speaker_loss_scale,
                embedding_dim=self.embedding_dim,
            )
            if num_classes is not None
            else None
        )

    def forward(
        self,
        *,
        waveforms: torch.Tensor,
        transcripts: Sequence[str],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Frame map: one acoustic sequence per utterance.
        frame_map = self.audio_encoder(
            waveforms,
            padding_mask=padding_mask,
            return_frame_features=True,
        )
        if "frame_map" in self.debug_pdb_stages:
            pdb.set_trace()
        # Token embeddings come directly from the LUAR input embedding table,
        # not from the full LUAR forward pass.
        text_embeddings, text_padding_mask = self._tokenize_and_embed(transcripts, waveforms.device)
        if "text_embeddings" in self.debug_pdb_stages:
            pdb.set_trace()
        # Each text token attends over the full acoustic frame sequence.
        aligned_audio = self.cross_attention(
            text_embeddings,
            frame_map,
            text_padding_mask=text_padding_mask,
        )
        if "aligned_audio" in self.debug_pdb_stages:
            pdb.set_trace()
        if self.fusion_mode == "concat":
            token_features = self.fusion_proj(torch.cat([text_embeddings.float(), aligned_audio], dim=-1))
        else:
            assert self.text_to_dv is not None
            assert self.fusion_layer_norm is not None
            token_features = self.fusion_proj(
                self.fusion_layer_norm(aligned_audio + self.text_to_dv(text_embeddings.float()))
            )
        if "token_features" in self.debug_pdb_stages:
            pdb.set_trace()

        # This stage lets already fused token representations exchange context
        # with one another before we collapse the sequence.
        token_features = self._apply_token_self_attention(token_features, text_padding_mask)
        hybrid_embedding = self._pool_token_sequence(token_features, text_padding_mask)
        if "hybrid_embedding" in self.debug_pdb_stages:
            pdb.set_trace()

        if self.use_luar_stream:
            if self.sluar_proj is None:
                raise RuntimeError("LUAR stream projection is missing.")
            # Separate from token-level cross-attention, this branch computes a
            # single utterance-level LUAR embedding from the full transcript.
            utterance_text = self.text_encoder(list(transcripts))
            utterance_text = utterance_text[0] if isinstance(utterance_text, tuple) else utterance_text
            utterance_text = self.sluar_proj(utterance_text.float())
            fused = self.final_proj(torch.cat([hybrid_embedding, utterance_text], dim=-1))
        else:
            fused = self.final_proj(hybrid_embedding)
        if "fused" in self.debug_pdb_stages:
            pdb.set_trace()

        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused

    def _resolve_token_embedding_table(self) -> nn.Module:
        if hasattr(self.text_encoder, "get_input_embeddings"):
            embeddings = self.text_encoder.get_input_embeddings()
            if embeddings is not None:
                return embeddings
        model = getattr(self.text_encoder, "model", None)
        if model is not None and hasattr(model, "transformer") and hasattr(model.transformer, "embeddings"):
            return model.transformer.embeddings.word_embeddings
        raise RuntimeError("HybridAudioTextModel requires a text encoder that exposes token input embeddings.")

    def _resolve_special_token_ids(self) -> set[int]:
        tokenizer = getattr(self.text_encoder, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("HybridAudioTextModel requires a text encoder tokenizer.")
        token_ids: set[int] = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                token_ids.add(int(value))
        return token_ids

    def _token_embedding_dim(self) -> int:
        if hasattr(self._token_embed_table, "embedding_dim"):
            return int(self._token_embed_table.embedding_dim)
        weight = getattr(self._token_embed_table, "weight", None)
        if weight is None:
            raise RuntimeError("Could not infer token embedding dimension.")
        return int(weight.shape[-1])

    def _tokenize_and_embed(
        self,
        transcripts: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.text_encoder.tokenizer
        encodings = tokenizer(
            list(transcripts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        if "input_ids" in self.debug_pdb_stages:
            pdb.set_trace()

        # We exclude both tokenizer padding and special tokens such as BOS/EOS
        # from the token pooling path.
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self._special_token_ids:
            special_mask |= input_ids == token_id
        text_padding_mask = (~attention_mask.bool()) | special_mask

        if any(parameter.requires_grad for parameter in self._token_embed_table.parameters()):
            text_embeddings = self._token_embed_table(input_ids)
        else:
            with torch.no_grad():
                text_embeddings = self._token_embed_table(input_ids)
        return text_embeddings, text_padding_mask

    def _apply_token_self_attention(
        self,
        token_features: torch.Tensor,
        text_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_token_self_attention:
            return token_features
        return self.token_self_attention(token_features, src_key_padding_mask=text_padding_mask)

    def _pool_token_sequence(
        self,
        token_features: torch.Tensor,
        text_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Mean pooling gives one utterance-level summary of the fused token
        # sequence. Optional stat pooling appends token-wise standard deviation.
        valid = (~text_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        mean = (token_features * valid).sum(dim=1) / denom
        if not self.use_stat_pooling:
            return mean
        mean_sq = ((token_features**2) * valid).sum(dim=1) / denom
        variance = (mean_sq - mean**2).clamp(min=1e-5)
        std = torch.sqrt(variance)
        return torch.cat([mean, std], dim=-1)


def build_hybrid_audio_text_model(config: ExperimentConfig) -> HybridAudioTextModel:
    """Build the hybrid acoustic-textual model described in the paper."""
    options = dict(config.model.options)
    speaker_loss = dict(options.pop("speaker_loss", {}) or {})
    audio_encoder_config = with_default_download_root(
        options.pop("audio_encoder", None),
        output_dir=config.output_dir,
        experiment_name=config.name,
    )
    text_encoder_config = with_default_download_root(
        options.pop("text_encoder", None),
        output_dir=config.output_dir,
        experiment_name=config.name,
    )
    audio_encoder = build_audio_encoder(audio_encoder_config)
    text_encoder = build_text_encoder(text_encoder_config)
    if audio_encoder is None:
        raise ValueError("hybrid_audio_text requires audio_encoder.")
    if text_encoder is None:
        raise ValueError("hybrid_audio_text requires text_encoder.")

    return HybridAudioTextModel(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        frame_dim=int(options.pop("frame_dim", 1536)),
        d_k=int(options.pop("d_k", 256)),
        d_v=int(options.pop("d_v", 256)),
        fusion_mode=str(options.pop("fusion_mode", "concat")),
        cross_attn_heads=int(options.pop("cross_attn_heads", 4)),
        self_attn_heads=int(options.pop("self_attn_heads", 4)),
        self_attn_layers=int(options.pop("self_attn_layers", 1)),
        use_token_self_attention=bool(options.pop("use_token_self_attention", True)),
        use_luar_stream=bool(options.pop("use_luar_stream", True)),
        use_stat_pooling=bool(options.pop("use_stat_pooling", False)),
        hybrid_dim=int(options.pop("hybrid_dim", 256)),
        final_dim=int(options.pop("final_dim", 256)),
        text_dim=_optional_int(options.pop("text_dim", None)),
        sluar_dim=int(options.pop("sluar_dim", 512)),
        normalize=bool(options.pop("normalize", True)),
        debug_pdb_stages=options.pop("debug_pdb_stages", None),
        num_classes=_optional_int(options.pop("num_classes", None)),
        speaker_loss_margin=float(speaker_loss.pop("margin", 0.2)),
        speaker_loss_scale=float(speaker_loss.pop("scale", 30.0)),
    )


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "CrossAttentionAligner",
    "HybridAudioTextModel",
    "build_hybrid_audio_text_model",
]
