from __future__ import annotations

import json

import torch
import torchaudio

from multimodal_asv.config import load_config
from multimodal_asv.registry import build_model
from scripts.eval_trials import _encode_side, _family_side_pooling, _load_checkpoint_model


def _write_wav(path, values: torch.Tensor, sample_rate: int = 16000) -> None:
    torchaudio.save(str(path), values.unsqueeze(0), sample_rate)


def test_eval_trials_loads_checkpoint_and_scores_mean_pooled_audio(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_mean_pool
model:
  family: mean_pooled_audio
  audio_encoder:
    type: identity
  embedding_dim: 1600
data:
  sample_rate: 16000
  segment_seconds: 0.1
training: {}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    model = build_model(config)
    ckpt_path = tmp_path / "epoch_001.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    loaded = _load_checkpoint_model(config, ckpt_path, torch.device("cpu"))

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_wav(wav_a, torch.ones(1600))
    _write_wav(wav_b, -torch.ones(1600))

    enroll = {"audio_paths": [str(wav_a), str(wav_a)]}
    same = {"audio_paths": [str(wav_a), str(wav_a)]}
    diff = {"audio_paths": [str(wav_b), str(wav_b)]}

    enroll_emb = _encode_side(
        loaded,
        config,
        enroll,
        trial_dir=tmp_path,
        transcript_index=None,
        device=torch.device("cpu"),
    )
    same_emb = _encode_side(
        loaded,
        config,
        same,
        trial_dir=tmp_path,
        transcript_index=None,
        device=torch.device("cpu"),
    )
    diff_emb = _encode_side(
        loaded,
        config,
        diff,
        trial_dir=tmp_path,
        transcript_index=None,
        device=torch.device("cpu"),
    )

    same_score = torch.nn.functional.cosine_similarity(enroll_emb, same_emb, dim=0).item()
    diff_score = torch.nn.functional.cosine_similarity(enroll_emb, diff_emb, dim=0).item()

    assert same_score > diff_score
    assert _family_side_pooling(config) == "model_grouped_episode"


def test_eval_trials_encodes_utterance_audio_text_by_mean_pooling(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_utt_audio_text
model:
  family: utterance_audio_text_fusion
  audio_encoder:
    type: identity
  text_encoder:
    type: static
    embedding_dim: 4
  fusion:
    modalities: [audio, text]
    audio_dim: 1600
    text_dim: 4
    projection_dim: 8
    final_dim: 5
data:
  sample_rate: 16000
  segment_seconds: 0.1
training: {}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    model = build_model(config)
    ckpt_path = tmp_path / "epoch_001.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    loaded = _load_checkpoint_model(config, ckpt_path, torch.device("cpu"))

    wav_a = tmp_path / "a.wav"
    _write_wav(wav_a, torch.linspace(0.0, 1.0, 1600))

    side = {
        "audio_paths": [str(wav_a), str(wav_a)],
        "transcripts": ["hello", "hello there"],
    }

    emb = _encode_side(
        loaded,
        config,
        side,
        trial_dir=tmp_path,
        transcript_index=None,
        device=torch.device("cpu"),
    )

    assert emb.shape == (5,)
    assert torch.allclose(emb.norm(dim=0), torch.tensor(1.0), atol=1e-6)
    assert _family_side_pooling(config) == "mean_pool_utterance_embeddings"


def test_eval_trials_trial_json_shape_example() -> None:
    row = {
        "label": 1,
        "enroll": {"audio_paths": ["enroll_1.wav", "enroll_2.wav"], "transcripts": ["a", "b"]},
        "test": {"audio_paths": ["test_1.wav"], "transcripts": ["c"]},
    }
    payload = json.dumps(row)
    parsed = json.loads(payload)

    assert parsed["label"] == 1
    assert len(parsed["enroll"]["audio_paths"]) == 2
    assert len(parsed["test"]["transcripts"]) == 1
