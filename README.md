# Multimodal Speaker Verification as a Threat to Speaker Anonymization

[![arXiv](https://img.shields.io/badge/arXiv-2607.19636-b31b1b.svg)](https://arxiv.org/abs/2607.19636)

This is the official repository for the paper **[“Multimodal Speaker Verification as a Threat to Speaker Anonymization”](https://arxiv.org/abs/2607.19636)**.


## Abstract 

>*Most automatic speaker verification (ASV) systems
operate on individual utterances, despite real-world interactions
typically consisting of multiple utterances. As speech accumu-
lates, increasingly rich speaker information becomes available
through acoustic, prosodic, and linguistic cues, potentially chal-
lenging speaker anonymization methods that primarily target
vocal characteristics. We investigate ASV in a multi-utterance,
multimodal setting and examine whether aggregating information
across anonymized speech impacts privacy. We first study audio-
only aggregation across multiple anonymized utterances and
observe consistent performance improvements as more speech
becomes available. We then incorporate prosodic and linguistic
information, showing that multimodal systems outperform uni-
modal approaches. Finally, we compare aggregation strategies
and find that frame-level aggregation yields the lowest EERs.
Even with only five anonymized utterances, combining audio
and text reduces EER by over 15% relative to audio-only ag-
gregation, demonstrating that substantial speaker-discriminative
information remains accessible despite anonymization*.

## Data

Following prior work on privacy in long-form speech, [*Content Anonymization for Privacy in Long-form Audio*](https://arxiv.org/abs/2510.12780), we use the Fisher English Training Speech corpus:

- [Part 1 — LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13)
- [Part 2 — LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)

Fisher comprises two-speaker telephone conversations with multiple utterances from each participant, making it well suited for our multi-utterance, multimodal setting. In addition to supporting a long-form, multi-utterance setting, Fisher’s conversational nature captures speaker-dependent linguistic patterns, including lexical preferences, phrasing, and discourse style, that may accumulate across a conversation and reveal additional speaker information.



> **Note:** The Fisher corpus is distributed under license by the [Linguistic Data Consortium (LDC)](https://www.ldc.upenn.edu/) and is not included in this repository. Users must obtain both parts directly from the LDC. 


### Anonymized Speech

We additionally use an anonymized version of the Fisher corpus generated using [Stream-Voice-Anon](https://arxiv.org/abs/2601.13948). Each Fisher utterance is anonymized independently using a target speaker randomly selected from the [`train-clean-360` and `train-other-500` subsets of LibriSpeech](https://www.openslr.org/12/). The target-speaker pool includes only LibriSpeech utterances longer than 4 seconds.



## Quick Start

This repository supports Python 3.10 and PyTorch 2.4.0.

### Environment Setup
Create the environment from the repository root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .[dev,whisper]
```

### Pretrained Checkpoints
WavLM-Large checkpoint:

- Download `WavLM-Large.pt` from [Google Drive](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view).
- Set `model.audio_encoder.wavlm_checkpoint` in the config to the downloaded file.
- Example: `/path/to/checkpoints/WavLM-Large.pt`

Whisper checkpoint:

- Whisper support is based on [openai/whisper](https://github.com/openai/whisper).
- The Whisper-backed model uses `model.whisper_encoder.checkpoint`.
- Use a model name such as `medium` with `allow_download: true`, or set it to a local checkpoint path and use `allow_download: false`.

Training configs are in `configs/train/`.

## Data Preparation

The repository includes data-preparation utilities under `scripts/data_prep/` for:

- creating speaker-level train/validation splits
- sampling validation speakers
- splitting validation audio into enroll and target subsets
- creating multi-utterance trial JSONL files
- creating aligned `N=5`, `N=10`, and `N=15` trial files where smaller trials are subsets of larger ones
- building manifests from generated trials

The expected starting point is a plain text file with one utterance per line:

```text
speaker_id /path/to/audio.wav
speaker_id /path/to/another_audio.wav
```

Use the scripts in the following order, depending on what you need.

### 1. Train/validation speaker split

If you need a reproducible speaker-level split from larger source manifests, first create train and validation speaker subsets.

Expected inputs:

- a train audio manifest
- a validation audio manifest

Outputs:

- train subset manifest
- validation subset manifest
- speaker lists for each split

This script will be added from the existing split-preparation workflow if you want the full train/validation split process in this repo.

### 2. Validation speaker subset

If you already have a validation manifest and want a fixed speaker subset for trial generation, sample speakers first.

Example workflow:

```bash
uv run python scripts/data_prep/sample_val_speakers.py \
  --input-file /path/to/all_val_data.txt \
  --output-file data/val_500spk.txt \
  --num-speakers 500 \
  --seed 42
```

This writes a speaker-filtered validation list such as `data/val_500spk.txt`.

### 3. Enroll/target split

Split the sampled validation subset into per-speaker enroll and target pools:

```bash
uv run python scripts/data_prep/split_enroll_target.py \
  --audio-subset data/val_500spk.txt \
  --enroll-file data/enroll.txt \
  --target-file data/target.txt \
  --stats-file data/split_stats.txt \
  --enroll-ratio 0.3 \
  --seed 42
```

Outputs:

- `data/enroll.txt`
- `data/target.txt`
- `data/split_stats.txt`

### 4. Create trial JSONL files

Create a higher-`N` trial file first, for example `N=15`:

```bash
uv run python scripts/data_prep/create_trial_jsonl.py \
  --enroll-file data/enroll.txt \
  --target-file data/target.txt \
  --output-file data/trials_N15.jsonl \
  --num-utts 15 \
  --trials-per-speaker 500 \
  --seed 42
```

This writes trial rows with:

- positive trials: same-speaker enroll/test pairs
- negative trials: different-speaker enroll/test pairs
- `15` utterances per side for each kept trial

### 5. Create aligned lower-`N` trial files

Create aligned smaller-`N` trial files that are deterministic subsets of the larger anchor trial file:

```bash
uv run python scripts/data_prep/align_trial_pairs_across_n.py \
  --anchor-jsonl data/trials_N15.jsonl \
  --n-values 5 10 15 \
  --selection random_nested \
  --seed 42
```

This step is what gives you matched `N=5`, `N=10`, and `N=15` trial files where the smaller trial sides are subsets of the larger ones.

### 6. Build manifests from trials

Build an audio-only manifest from one or more generated trial JSONLs:

```bash
uv run python scripts/data_prep/build_audio_only_manifest_from_trials.py \
  --trial-dir data \
  --trial-glob 'trials*.jsonl' \
  --manifest-out data/audio_only_manifest.txt
```

### Current scope

The repository currently includes the validation subset, trial, and manifest preparation steps above.

Still to be added:

- full train/validation speaker split scripts
- LUAR split preparation from audio speaker lists
- audio+text validation asset preparation
- anonymized-data split and manifest generation

Train with:

```bash
uv run python scripts/train.py --config <config.yaml>
```

Validate a config before launching training:

```bash
uv run python scripts/train.py --config configs/train/audio_only_query_attention.yaml --dry-run
```

Manifest formats:

Audio-only:

```text
speaker_id /path/to/audio.wav
speaker_id /path/to/another_audio.wav
```

Audio plus text:

```text
speaker_id /path/to/audio.wav transcript text goes here
speaker_id /path/to/another_audio.wav another transcript
```

For audio-text and hybrid training, everything after the audio path is treated as transcript text. Grouped multi-utterance families still use one utterance per row; the dataloader samples multiple utterances for the same speaker within each episode.

Training workflow:

1. Edit the paper config under `configs/train/`.
2. Set manifest, checkpoint, and output paths.
3. Run `--dry-run`.
4. Launch training with the same config.

Run text-only LUAR evaluation with:

```bash
uv run python scripts/eval_text_only_trials.py \
  --trials-jsonl /path/to/text_only_trials.jsonl \
  --checkpoint-path /path/to/luar_checkpoint.ckpt \
  --score-out runs/text_only/scores.pt \
  --summary-out runs/text_only/summary.json
```

Run audio or multimodal checkpoint evaluation with:

```bash
uv run python scripts/eval_trials.py \
  --config configs/train/frame_audio_text_concat.yaml \
  --checkpoint-path /path/to/epoch_005.pt \
  --trials-jsonl /path/to/trials.jsonl \
  --score-out runs/frame_audio_text/scores.pt \
  --summary-out runs/frame_audio_text/summary.json
```

Trial JSONL formats:

Text-only:

```json
{"label": 1, "enroll": {"transcripts": ["utt one", "utt two"]}, "test": {"transcripts": ["utt three"]}}
```

Audio or multimodal:

```json
{"label": 1, "enroll": {"audio_paths": ["audio/spk1_1.wav", "audio/spk1_2.wav"], "transcripts": ["utt one", "utt two"]}, "test": {"audio_paths": ["audio/spk1_3.wav"], "transcripts": ["utt three"]}}
```

For audio-only evaluation, omit transcript lists. Relative audio paths are resolved relative to the trial JSONL file. For prosody-based evaluation, the config must provide `data.prosody_transcript_path`.

## Documentation

- [docs/data_formats.md](docs/data_formats.md) for manifest and trial file formats

## Citation

If you use this pipeline, please cite:

```bibtex
@article{garg2026multimodal,
  title={Multimodal Speaker Verification as a Threat to Speaker Anonymization},
  author={Garg, Ashi and Aggazzotti, Cristina and Garc{\'\i}a-Perera, Leibny Paola and Andrews, Nicholas},
  journal={arXiv preprint arXiv:2607.19636},
  year={2026}
}
```
