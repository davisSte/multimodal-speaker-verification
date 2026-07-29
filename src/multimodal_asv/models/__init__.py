"""Model family implementations."""

from .mean_pooled_audio import (
    MeanPooledAudioModel,
    build_mean_pooled_audio_model,
)
from .frame_pooled_audio_prosody import (
    FramePooledAudioProsodyModel,
    build_frame_pooled_audio_prosody_model,
)
from .frame_pooled_audio import (
    FramePooledAudioModel,
    MeanStdFramePooler,
    build_frame_pooled_audio_model,
)
from .frame_pooled_audio_text import (
    FramePooledAudioTextModel,
    WeightedSumFusion,
    build_frame_pooled_audio_text_model,
)
from .rjca_audio_global_luar import (
    RJCAGlobalRepeatAudioLuarModel,
    build_rjca_audio_global_luar_model,
)
from .hybrid_audio_text import (
    HybridAudioTextModel,
    build_hybrid_audio_text_model,
)
from .query_attention_audio import (
    QueryAttentionAggregation,
    QueryAttentionAudioModel,
    build_query_attention_audio_model,
)
from .utterance_audio_text_fusion import (
    ProjectedConcatFusion,
    UtteranceAudioTextFusionModel,
    build_utterance_audio_text_fusion_model,
)
from .whisper_wavlm_cross_attention import (
    CrossAttentionAligner,
    WhisperSequenceEncoder,
    WhisperWavlmCrossAttentionModel,
    build_whisper_wavlm_cross_attention_model,
)

__all__ = [
    "CrossAttentionAligner",
    "FramePooledAudioModel",
    "FramePooledAudioProsodyModel",
    "FramePooledAudioTextModel",
    "HybridAudioTextModel",
    "MeanPooledAudioModel",
    "MeanStdFramePooler",
    "ProjectedConcatFusion",
    "QueryAttentionAggregation",
    "QueryAttentionAudioModel",
    "RJCAGlobalRepeatAudioLuarModel",
    "UtteranceAudioTextFusionModel",
    "WeightedSumFusion",
    "WhisperSequenceEncoder",
    "WhisperWavlmCrossAttentionModel",
    "build_mean_pooled_audio_model",
    "build_frame_pooled_audio_model",
    "build_frame_pooled_audio_prosody_model",
    "build_frame_pooled_audio_text_model",
    "build_hybrid_audio_text_model",
    "build_query_attention_audio_model",
    "build_rjca_audio_global_luar_model",
    "build_utterance_audio_text_fusion_model",
    "build_whisper_wavlm_cross_attention_model",
]
