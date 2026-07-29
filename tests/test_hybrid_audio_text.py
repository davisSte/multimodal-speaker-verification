from __future__ import annotations

import torch
from torch import nn

from multimodal_asv.config import load_config
from multimodal_asv.models import HybridAudioTextModel
from multimodal_asv.registry import build_model


class TinyTokenizer:
    def __init__(self) -> None:
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.bos_token_id = None
        self.eos_token_id = None
        self.vocab = {
            "<s>": 0,
            "</s>": 1,
            "<pad>": 2,
            "hello": 3,
            "speaker": 4,
            "another": 5,
            "voice": 6,
        }

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        del padding, truncation, return_tensors
        rows = []
        max_len = 0
        for text in texts:
            tokens = [self.cls_token_id]
            tokens.extend(self.vocab.get(word, 3) for word in text.split())
            tokens.append(self.sep_token_id)
            rows.append(tokens)
            max_len = max(max_len, len(tokens))
        input_ids = []
        attention_mask = []
        for tokens in rows:
            pad = [self.pad_token_id] * (max_len - len(tokens))
            input_ids.append(tokens + pad)
            attention_mask.append([1] * len(tokens) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TinyTextEncoder(nn.Module):
    def __init__(self, token_dim: int = 8, utterance_dim: int = 6) -> None:
        super().__init__()
        self.tokenizer = TinyTokenizer()
        self.token_embeddings = nn.Embedding(7, token_dim)
        self.utterance_proj = nn.Linear(token_dim, utterance_dim)

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embeddings

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(texts)
        input_ids = enc["input_ids"].to(self.token_embeddings.weight.device)
        token_emb = self.token_embeddings(input_ids)
        pooled = token_emb.mean(dim=1)
        return self.utterance_proj(pooled)


class TinyFrameAudioEncoder(nn.Module):
    def __init__(self, frame_dim: int = 4, frames: int = 5) -> None:
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.frames = int(frames)
        self.proj = nn.Linear(1, self.frame_dim)

    def forward(
        self,
        waveforms: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        return_frame_features: bool = False,
    ) -> torch.Tensor:
        del padding_mask
        mean = waveforms.mean(dim=1, keepdim=True)
        frame = self.proj(mean).unsqueeze(-1).expand(-1, -1, self.frames)
        if return_frame_features:
            return frame
        return frame.mean(dim=-1)


def test_hybrid_audio_text_forward_shape() -> None:
    model = HybridAudioTextModel(
        audio_encoder=TinyFrameAudioEncoder(frame_dim=4, frames=5),
        text_encoder=TinyTextEncoder(token_dim=8, utterance_dim=6),
        frame_dim=4,
        d_k=8,
        d_v=8,
        hybrid_dim=8,
        final_dim=5,
        text_dim=8,
        sluar_dim=6,
        use_token_self_attention=True,
        use_luar_stream=True,
        num_classes=2,
    )

    embeddings = model(
        waveforms=torch.randn(3, 160),
        transcripts=["hello speaker", "another voice", "hello another"],
    )

    assert embeddings.shape == (3, 5)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_hybrid_audio_text_builder_from_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_hybrid_audio_text
model:
  family: hybrid_audio_text
  audio_encoder:
    type: stub
  text_encoder:
    type: stub
  frame_dim: 4
  d_k: 8
  d_v: 8
  hybrid_dim: 8
  final_dim: 5
  text_dim: 8
  sluar_dim: 6
data: {}
training: {}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "multimodal_asv.models.hybrid_audio_text.build_audio_encoder",
        lambda config: TinyFrameAudioEncoder(frame_dim=4, frames=5),
    )
    monkeypatch.setattr(
        "multimodal_asv.models.hybrid_audio_text.build_text_encoder",
        lambda config: TinyTextEncoder(token_dim=8, utterance_dim=6),
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, HybridAudioTextModel)
