from __future__ import annotations

import torch

from multimodal_asv.config import load_config
from multimodal_asv.models import FramePooledAudioProsodyModel, MeanStdFramePooler
from multimodal_asv.registry import build_model


def test_frame_pooled_audio_prosody_fuses_grouped_stats() -> None:
    from multimodal_asv.models.frame_pooled_audio import FramePooledAudioModel

    audio_model = FramePooledAudioModel(
        frame_pooler=MeanStdFramePooler(frame_dim=4, embedding_dim=6),
        embedding_dim=6,
    )
    model = FramePooledAudioProsodyModel(
        audio_model=audio_model,
        prosody_dim=3,
        prosody_projection_dim=4,
        embedding_dim=6,
    )
    frame_features = torch.randn(3, 5, 4, 7)
    pitch_features_grouped = torch.randn(3, 5, 3)

    speaker_embeddings = model(
        frame_features=frame_features,
        pitch_features_grouped=pitch_features_grouped,
    )

    assert speaker_embeddings.shape == (3, 6)
    assert torch.allclose(speaker_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_frame_pooled_audio_prosody_builder_from_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: tiny_frame_pooled_audio_prosody
model:
  family: frame_pooled_audio_prosody
  embedding_dim: 6
  prosody:
    feature_dim: 3
    projection_dim: 4
  frame_pooler:
    type: mean_std
    frame_dim: 4
data: {}
training: {}
""",
        encoding="utf-8",
    )

    model = build_model(load_config(config_path))

    assert isinstance(model, FramePooledAudioProsodyModel)


def test_frame_pooled_audio_prosody_loads_released_checkpoint_key_names() -> None:
    from multimodal_asv.models.frame_pooled_audio import FramePooledAudioModel

    model = FramePooledAudioProsodyModel(
        audio_model=FramePooledAudioModel(
            frame_pooler=MeanStdFramePooler(frame_dim=4, embedding_dim=6),
            embedding_dim=6,
        ),
        prosody_dim=3,
        prosody_projection_dim=4,
        embedding_dim=6,
    )
    state = model.state_dict()
    released_style_state = {}
    for key, value in state.items():
        if key.startswith("audio_model.audio_encoder."):
            key = key.replace("audio_model.audio_encoder.", "audio_encoder.", 1)
        elif key.startswith("prosody_proj."):
            key = key.replace("prosody_proj.", "pitch_proj.", 1)
        elif key.startswith("prosody_pre_concat_ln."):
            key = key.replace("prosody_pre_concat_ln.", "pitch_pre_concat_ln.", 1)
        released_style_state[key] = value

    model.load_state_dict(released_style_state, strict=True)
