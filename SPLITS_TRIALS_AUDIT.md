# Splits, Trials, And Manifest Audit

This note summarizes what already exists in `hybrid-test/splits/scripts` and what still needs to be cleaned up before adding the code to this repository.

## Existing Scripts

### Speaker subset and split creation

- `sample_val_speakers.py`
  - samples a fixed number of validation speakers from a master speaker/audio list
  - keeps all utterances for each selected speaker

- `split_enroll_target.py`
  - splits each selected speaker's utterances into enroll and target subsets
  - writes:
    - enroll subset text file
    - target subset text file
    - per-speaker split statistics

### Trial generation

- `create_trial_jsonl.py`
  - creates positive and negative multi-utterance trial JSONL rows from enroll/target subsets
  - supports:
    - configurable `N` utterances per side
    - configurable trials per speaker
    - path-prefix remapping from one filesystem layout to another
    - writing remapped subset files for inspection

- `align_trial_pairs_across_n.py`
  - creates aligned trial JSONLs across multiple `N` values
  - preserves the same enroll/test speaker pairs across settings such as `N=5`, `N=10`, and `N=15`
  - supports deterministic nested random subset selection

### Manifest creation

- `build_audio_only_manifest_from_trials.py`
  - scans trial JSONLs
  - extracts unique audio paths and speaker IDs
  - writes an audio-only training/evaluation manifest

- `build_audio_text_val_assets.py`
  - augments audio-only trial JSONLs with transcripts and word timestamps
  - writes:
    - audio+text trial JSONLs
    - soft manifests
    - hard-inline manifests
    - summary reports

### Pipeline wrapper

- `build_val_trials_pipeline.py`
  - runs:
    - speaker sampling
    - enroll/target splitting
    - multi-`N` trial creation
  - useful as a reference for an end-to-end public pipeline

## What We Already Have

We already have local code for:

- validation speaker sampling
- enroll/target splitting
- multi-utterance trial generation
- aligned multi-`N` trial generation
- audio-only manifest creation from trials
- audio+text validation asset generation from transcript shards

## Main Cleanup Needed Before Porting

### Path cleanup

- some scripts still assume local filesystem remapping via:
  - `--path-prefix-from`
  - `--path-prefix-to`
- these should become either:
  - optional arguments with neutral defaults
  - or a separate path-rewrite utility

### Public-facing structure

- the scripts are currently standalone utilities under `hybrid-test/splits/scripts`
- for this repo they should probably become either:
  - `scripts/data_prep/*.py`
  - or package modules under `src/multimodal_asv/data_prep/`

### Code quality cleanup

- `split_enroll_target.py` needs the most cleanup
  - minimal docstrings
  - plain `open(...)` usage instead of `Path` methods
  - debug prints
  - little validation around malformed input

- `build_val_trials_pipeline.py` currently reuses other scripts by rewriting `sys.argv`
  - better public version would import callable functions directly

- `create_trial_jsonl.py` and `build_audio_text_val_assets.py` are the strongest starting points
  - these look closest to reusable public utilities

### Data assumptions that need documentation

- expected format of the master speaker/audio list
- how validation speakers are chosen
- how enroll/target ratio is defined
- how negative speakers are sampled
- how transcript shard JSONLs are structured
- what "soft" and "hard_inline" manifests mean

## Recommended Port Order

1. `create_trial_jsonl.py`
2. `align_trial_pairs_across_n.py`
3. `build_audio_only_manifest_from_trials.py`
4. `sample_val_speakers.py`
5. `split_enroll_target.py`
6. `build_audio_text_val_assets.py`
7. `build_val_trials_pipeline.py`

## Missing Beyond These Scripts

These scripts cover validation splits, trials, and some manifest generation, but they do not yet cover everything we still need:

- full Fisher train/dev/eval split preparation
- anonymized-data split and manifest generation
- paper-final trial generation recipe documentation
- release-ready example inputs for smoke testing
