# Multimodal Speaker Verification as a Threat to Speaker Anonymization

[![arXiv](https://img.shields.io/badge/arXiv-2607.19636-b31b1b.svg)](https://arxiv.org/abs/2607.19636)

Official implementation of **[“Multimodal Speaker Verification as a Threat to Speaker Anonymization”](https://arxiv.org/abs/2607.19636)**..

## Abstract

>*Most automatic speaker verification (ASV) systems
operate on individual utterances, despite real-world interactions
typically consisting of multiple utterances. As speech accumulates,
increasingly rich speaker information becomes available
through acoustic, prosodic, and linguistic cues, potentially challenging
speaker anonymization methods that primarily target
vocal characteristics. We investigate ASV in a multi-utterance,
multimodal setting and examine whether aggregating information
across anonymized speech impacts privacy. We first study audio-only
aggregation across multiple anonymized utterances and
observe consistent performance improvements as more speech
becomes available. We then incorporate prosodic and linguistic
information, showing that multimodal systems outperform unimodal
approaches. Finally, we compare aggregation strategies
and find that frame-level aggregation yields the lowest EERs.
Even with only five anonymized utterances, combining audio
and text reduces EER by over 15% relative to audio-only aggregation,
demonstrating that substantial speaker-discriminative
information remains accessible despite anonymization*.

## Quick Start

This repository supports Python 3.10 and PyTorch 2.4.0.

### 1. Create the environment

Run the following commands from the repository root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[dev,whisper]
```

Install anonymization dependencies only when running StreamVoiceAnon inference:

```bash
uv pip install -e .[anon]
```

### 2. Download pretrained checkpoints

#### Released ASV checkpoints

Pretrained checkpoints for the paper models are hosted on [Hugging Face](https://huggingface.co/ash56/multimodal-asv-checkpoints).

Evaluate directly from Hugging Face paths:

```bash
uv run python scripts/eval_trials.py \
  --config hf://ash56/multimodal-asv-checkpoints/audio_only/query_attention/config.yaml \
  --checkpoint-path hf://ash56/multimodal-asv-checkpoints/audio_only/query_attention/checkpoint.pt \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --trials-jsonl /path/to/trials.jsonl \
  --score-out runs/audio_only_query_attention/scores.pt \
  --summary-out runs/audio_only_query_attention/summary.json
```

Or download the checkpoint and matching evaluation config first:

```bash
hf download ash56/multimodal-asv-checkpoints \
  audio_text/rjca_global_luar/checkpoint.pt \
  audio_text/rjca_global_luar/config.yaml \
  --local-dir checkpoints
```

Then evaluate from the local files:

```bash
uv run python scripts/eval_trials.py \
  --config checkpoints/audio_text/rjca_global_luar/config.yaml \
  --checkpoint-path checkpoints/audio_text/rjca_global_luar/checkpoint.pt \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --text-encoder-checkpoint /path/to/luar_checkpoint.ckpt \
  --trials-jsonl /path/to/trials.jsonl \
  --score-out runs/rjca_global_luar/scores.pt \
  --summary-out runs/rjca_global_luar/summary.json
```

Use the `config.yaml` downloaded with each checkpoint for evaluation. See the Hugging Face repository for the current list of available models.

To load a released checkpoint directly in Python and score two multi-utterance sides:

```python
from pathlib import Path

import torch
import torch.nn.functional as F

from multimodal_asv.config import load_config
from multimodal_asv.evaluation.trials import encode_trial_side, load_checkpoint_model
from multimodal_asv.utils.huggingface import resolve_hf_or_local_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = resolve_hf_or_local_path(
    "hf://ash56/multimodal-asv-checkpoints/audio_only/query_attention/config.yaml"
)
checkpoint_path = resolve_hf_or_local_path(
    "hf://ash56/multimodal-asv-checkpoints/audio_only/query_attention/checkpoint.pt"
)

config = load_config(config_path)
config.model.options["audio_encoder"]["wavlm_checkpoint"] = "/path/to/WavLM-Large.pt"

model = load_checkpoint_model(config, checkpoint_path, device)

enroll = {"audio_paths": ["speaker_a_utt1.wav", "speaker_a_utt2.wav"]}
test = {"audio_paths": ["candidate_utt1.wav", "candidate_utt2.wav"]}

enroll_embedding = encode_trial_side(
    model,
    config,
    enroll,
    trial_dir=Path("."),
    transcript_index=None,
    device=device,
)
test_embedding = encode_trial_side(
    model,
    config,
    test,
    trial_dir=Path("."),
    transcript_index=None,
    device=device,
)

score = F.cosine_similarity(enroll_embedding, test_embedding, dim=0).item()
print(score)
```

For text or audio-text models, also set the local `model.text_encoder.checkpoint_path` required by that model's config.

#### WavLM-Large

1. Download `WavLM-Large.pt` from [Google Drive](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view).
2. Set `model.audio_encoder.wavlm_checkpoint` in the selected configuration to the downloaded checkpoint path.

#### Whisper

Whisper support is based on [openai/whisper](https://github.com/openai/whisper). The Whisper-backed model uses `model.whisper_encoder.checkpoint`.

Use a model name such as `medium` with `allow_download: true`, or provide a local checkpoint path with `allow_download: false`.

#### StreamVoiceAnon

The StreamVoiceAnon source is included under:

```text
src/multimodal_asv/anonymization/StreamVoiceAnon
```

The checkpoint is not included. See [Data Preparation](docs/data_preparation.md#11-prepare-streamvoiceanon) for the download command.

### 3. Configure local paths

Select a configuration under `configs/train/` and set the required:

- training and validation manifests
- pretrained checkpoint paths
- output directory
- modality-specific inputs, when applicable

### 4. Validate the configuration

Before starting training, run a dry run:

```bash
uv run python scripts/train.py \
  --config configs/train/audio_only_query_attention.yaml \
  --dry-run
```

### 5. Train or evaluate

Train with:

```bash
uv run python scripts/train.py \
  --config configs/train/audio_only_query_attention.yaml
```

Evaluate an audio or multimodal checkpoint with:

```bash
uv run python scripts/eval_trials.py \
  --config configs/train/frame_audio_text_concat.yaml \
  --checkpoint-path /path/to/epoch_005.pt \
  --trials-jsonl /path/to/trials.jsonl \
  --score-out runs/frame_audio_text/scores.pt \
  --summary-out runs/frame_audio_text/summary.json
```

See [Training](#training) and [Evaluation](#evaluation) for the complete workflow.

## Data

### Fisher corpus

Following prior work on privacy in long-form speech, [*Content Anonymization for Privacy in Long-form Audio*](https://arxiv.org/abs/2510.12780), the experiments use the Fisher English Training Speech corpus:

- [Part 1 speech: LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13)
- [Part 2 speech: LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)
- [Part 1 transcripts: LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19)
- [Part 2 transcripts: LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19)

Fisher contains two-speaker telephone conversations with multiple utterances from each participant. This supports the multi-utterance setting while also providing conversational text from which speaker-dependent lexical, phrasing, and discourse patterns may accumulate.

> **License:** Fisher is distributed by the [Linguistic Data Consortium](https://www.ldc.upenn.edu/) and is not included in this repository. Users must obtain the speech and transcript releases directly from the LDC.

### Anonymized speech

The paper additionally uses Fisher speech anonymized with [StreamVoiceAnon](https://arxiv.org/abs/2601.13948). Each Fisher utterance is anonymized independently using a target utterance randomly sampled from the LibriSpeech [`train-clean-360` and `train-other-500` subsets](https://www.openslr.org/12/). Only reference utterances longer than four seconds are included in the target-speaker pool.

Complete Fisher preparation, trial creation, and anonymization commands are in [docs/data_preparation.md](docs/data_preparation.md).

## Training

Training configurations are stored under `configs/train/`.

### Manifest formats

Audio-only manifests contain one utterance per row:

```text
speaker_id /path/to/audio.wav
speaker_id /path/to/another_audio.wav
```

Audio-plus-text manifests append the transcript after the audio path:

```text
speaker_id /path/to/audio.wav transcript text goes here
speaker_id /path/to/another_audio.wav another transcript
```

For audio-text and hybrid training, everything after the audio path is treated as transcript text. Multi-utterance training still uses one utterance per row; the dataloader samples multiple utterances for the same speaker within each episode.

For full format specifications, see [docs/data_formats.md](docs/data_formats.md).

### Training workflow

1. Select a paper configuration under `configs/train/`.
2. Set the manifest, checkpoint, and output paths.
3. Validate the configuration with `--dry-run`.
4. Launch training with the same configuration.

Validate a configuration:

```bash
uv run python scripts/train.py \
  --config configs/train/audio_only_query_attention.yaml \
  --dry-run
```

Launch training:

```bash
uv run python scripts/train.py --config <config.yaml>
```

## Evaluation

### Audio-only and multimodal evaluation

```bash
uv run python scripts/eval_trials.py \
  --config configs/train/frame_audio_text_concat.yaml \
  --checkpoint-path /path/to/epoch_005.pt \
  --trials-jsonl /path/to/trials.jsonl \
  --score-out runs/frame_audio_text/scores.pt \
  --summary-out runs/frame_audio_text/summary.json
```

An audio or multimodal trial has the following structure:

```json
{"label": 1, "enroll": {"audio_paths": ["audio/spk1_1.wav", "audio/spk1_2.wav"], "transcripts": ["utt one", "utt two"]}, "test": {"audio_paths": ["audio/spk1_3.wav"], "transcripts": ["utt three"]}}
```

For audio-only evaluation, omit the transcript lists. Relative audio paths are resolved relative to the trial JSONL file. For prosody-based evaluation, the configuration must provide `data.prosody_transcript_path`.

### Text-only LUAR evaluation

```bash
uv run python scripts/eval_text_only_trials.py \
  --trials-jsonl /path/to/text_only_trials.jsonl \
  --checkpoint-path /path/to/luar_checkpoint.ckpt \
  --score-out runs/text_only/scores.pt \
  --summary-out runs/text_only/summary.json
```

A text-only trial has the following structure:

```json
{"label": 1, "enroll": {"transcripts": ["utt one", "utt two"]}, "test": {"transcripts": ["utt three"]}}
```

For complete trial schemas, see [docs/data_formats.md](docs/data_formats.md).

### Closed-set re-identification and LID

EER evaluates pairwise verification and does not directly measure a 1-to-N identity-linkage
attack. The project-local [LID evaluator](lid/README.md) reports Local Information Disclosure
(ALID, PDR/NDR, conditional and maximum LID) together with rank-1, rank-5, and mean reciprocal
rank. It requires exhaustive development and test score matrices with disjoint identities.

```bash
uv run python lid/evaluate.py \
  --dev-scores /path/to/dev_scores.csv \
  --test-scores /path/to/test_scores.csv \
  --output-dir runs/my_system/lid
```

## Documentation

- [docs/data_preparation.md](docs/data_preparation.md): Fisher preparation, trial generation, and StreamVoiceAnon anonymization
- [docs/data_formats.md](docs/data_formats.md): manifest and trial file formats

## Citation

Please cite this work when using the pipeline:

```bibtex
@article{garg2026multimodal,
  title={Multimodal Speaker Verification as a Threat to Speaker Anonymization},
  author={Garg, Ashi and Aggazzotti, Cristina and Garc{\'\i}a-Perera, Leibny Paola and Andrews, Nicholas},
  journal={arXiv preprint arXiv:2607.19636},
  year={2026}
}
```

**Disclaimer:** The code was organized and cleaned up for public release with the help of OpenAI Codex.

If you find bugs or inconsistencies, please open an issue or submit a pull request.
