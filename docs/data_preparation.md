# Data Preparation

This guide describes how to prepare Fisher utterance manifests, validation trials, audio-text assets, and anonymized Fisher audio for the experiments.

The data-preparation utilities are under:

```text
scripts/data_prep/
```

The anonymization utilities are Python modules under:

```text
multimodal_asv.anonymization
```

## Fisher Releases

The experiments use Fisher English Training Speech:

- [Part 1 speech: LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13)
- [Part 2 speech: LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13)
- [Part 1 transcripts: LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19)
- [Part 2 transcripts: LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19)

The Fisher corpus is distributed by the [Linguistic Data Consortium](https://www.ldc.upenn.edu/) and is not included in this repository. Users must obtain the speech and transcript releases directly from the LDC.

## 1. Segment Fisher Conversations

Fisher is distributed as two-speaker conversation audio. It must first be split into utterance-level wav files using transcript timestamps and speaker metadata. This follows the Fisher preparation used by [caggazzotti/long-form-speech-anonymization](https://github.com/caggazzotti/long-form-speech-anonymization).

Required inputs:

- Fisher speech audio from `LDC2004S13` and `LDC2005S13`
- Fisher transcripts from `LDC2004T19` and `LDC2005T19`
- Fisher speaker metadata tables, usually named `fe_03_pindata.tbl`
- `sph2pipe` for reading Fisher SPH audio
- an output directory for utterance wav files

The transcript roots must contain utterance start and end times, speaker-channel labels, and transcript text. The speaker-map files associate Fisher call and channel metadata with speaker IDs.

```bash
python /path/to/long-form-speech-anonymization/scripts/prepare_utterance_audio.py \
  --audio-root /path/to/LDC2004S13/audio \
  --audio-root /path/to/LDC2005S13/audio \
  --transcript-root /path/to/LDC2004T19/fe_03_p1_tran/data/trans \
  --transcript-root /path/to/LDC2005T19/data/trans \
  --speaker-map /path/to/LDC2004T19/fe_03_p1_tran/doc/fe_03_pindata.tbl \
  --speaker-map /path/to/LDC2005T19/doc/fe_03_pindata.tbl \
  --sph2pipe /path/to/sph2pipe \
  --output-dir data/fisher_utterance_audio
```

Expected output structure:

```text
data/fisher_utterance_audio/
  00001/
    fe_03_00001_A_0_2602.wav
    fe_03_00001_A_1_2602.wav
    fe_03_00001_B_0_9182.wav
  00002/
    fe_03_00002_A_0_1746.wav
```

The final underscore-separated field in each filename is the Fisher speaker ID.

## 2. Build the Utterance Manifest

```bash
uv run python scripts/data_prep/build_fisher_manifest_from_utterance_dir.py \
  --utterance-root data/fisher_utterance_audio \
  --manifest-out data/fisher_audio_manifest.txt \
  --metadata-out data/fisher_audio_manifest.metadata.json
```

The resulting manifest contains one utterance per line:

```text
speaker_id /path/to/audio.wav
speaker_id /path/to/another_audio.wav
```

## 3. Create Train and Validation Speaker Splits

Use this step when preparing reproducible speaker-level subsets from larger source manifests.

```bash
uv run python scripts/data_prep/split_audio_manifest_by_speaker.py \
  --train-source /path/to/train_audio_manifest.txt \
  --val-source /path/to/validation_audio_manifest.txt \
  --output-dir data/splits/fisher_audio \
  --num-train-speakers 2000 \
  --num-val-speakers 0 \
  --seed 42
```

Use `--num-val-speakers 0` to keep all validation speakers. The script verifies that train and validation speakers are disjoint and writes subset manifests, remaining-speaker manifests, speaker lists, and `split_metadata.json`.

## 4. Prepare the LUAR Text-Only Split

Create LUAR-style train and validation files from audio-text manifests using the speaker lists generated above:

```bash
uv run python scripts/data_prep/build_luar_split_from_audio_speakers.py \
  --train-source /path/to/train_audio_text_manifest.txt \
  --val-source /path/to/validation_audio_text_manifest.txt \
  --train-speakers data/splits/fisher_audio/train_speakers.txt \
  --val-speakers data/splits/fisher_audio/validation_speakers.txt \
  --dataset-name fisher \
  --output-dir data/splits/fisher_luar \
  --encoding ascii
```

The script accepts either of the following row formats:

```text
utterance_id|audio_path|transcript|speaker_id
speaker_id audio_path transcript text
```

It writes `train.jsonl`, `val.jsonl`, `validation_queries.jsonl`, `validation_targets.jsonl`, speaker maps, and `metadata.json`.

## 5. Select Validation Speakers

If you already have a validation manifest and need a fixed speaker subset for trial generation, sample it with:

```bash
uv run python scripts/data_prep/sample_val_speakers.py \
  --input-file /path/to/all_val_data.txt \
  --output-file data/val_500spk.txt \
  --num-speakers 500 \
  --seed 42
```

This writes a speaker-filtered validation list such as `data/val_500spk.txt`.

## 6. Create Enroll and Target Pools

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

## 7. Create the Anchor Trial File

Create the largest trial set first, for example `N=15`:

```bash
uv run python scripts/data_prep/create_trial_jsonl.py \
  --enroll-file data/enroll.txt \
  --target-file data/target.txt \
  --output-file data/trials_N15.jsonl \
  --num-utts 15 \
  --trials-per-speaker 500 \
  --seed 42
```

Each retained trial contains 15 enrollment utterances and 15 test utterances. Positive trials pair the same speaker, while negative trials pair different speakers.

## 8. Create Aligned Lower-N Trial Files

Generate deterministic nested subsets from the `N=15` anchor file:

```bash
uv run python scripts/data_prep/align_trial_pairs_across_n.py \
  --anchor-jsonl data/trials_N15.jsonl \
  --n-values 5 10 15 \
  --selection random_nested \
  --seed 42
```

This produces matched `N=5`, `N=10`, and `N=15` trial files in which the smaller trial sides are subsets of the larger ones.

## 9. Attach Transcripts to Validation Trials

```bash
uv run python scripts/data_prep/build_audio_text_val_assets.py \
  --trial-dir data \
  --trial-glob 'trials*.jsonl' \
  --transcription-dir /path/to/whisper_jsonl_shards \
  --out-dir data/audio_text_val
```

The transcription JSONL rows must include:

- a filename field such as `file`, `audio`, or `path`
- a `text` field
- optionally, Whisper-style `segments[*].words`

The script writes audio-text trial JSONLs, soft text manifests, hard-inline word-timestamp manifests, and a report of missing transcripts.

## 10. Build Manifests from Trials

Build an audio-only manifest from one or more generated trial files:

```bash
uv run python scripts/data_prep/build_audio_only_manifest_from_trials.py \
  --trial-dir data \
  --trial-glob 'trials*.jsonl' \
  --manifest-out data/audio_only_manifest.txt
```

## 11. Prepare StreamVoiceAnon

Install the optional anonymization dependencies:

```bash
uv pip install -e .[anon]
```

The bundled StreamVoiceAnon source is located under:

```text
src/multimodal_asv/anonymization/StreamVoiceAnon
```

The StreamVoiceAnon checkpoint is not included. Download the pretrained artifacts from [Plachta/StreamVoiceAnon on Hugging Face](https://huggingface.co/Plachta/StreamVoiceAnon):

```bash
mkdir -p src/multimodal_asv/anonymization/StreamVoiceAnon/pretrained_checkpoints
hf download Plachta/StreamVoiceAnon \
  --local-dir src/multimodal_asv/anonymization/StreamVoiceAnon/pretrained_checkpoints
```

The main checkpoint used by the wrapper is expected at:

```text
src/multimodal_asv/anonymization/StreamVoiceAnon/pretrained_checkpoints/dual_ar_delay_0_8.pth
```

Alternatively, store the checkpoint elsewhere and pass its absolute path with `--checkpoint-path`.

## 12. Create Source-Reference Pairs

Anonymization uses a source-reference pair file with one conversion per line:

```text
source_wav|source_speaker|reference_wav|reference_speaker
```

For example:

```text
/path/to/fisher/utt1.wav|fisher_spk1|/path/to/librispeech/ref1.flac|libri_spk1
/path/to/fisher/utt2.wav|fisher_spk1|/path/to/librispeech/ref2.flac|libri_spk2
```

Create this file from a Fisher source manifest and a LibriSpeech reference root:

```bash
uv run python -m multimodal_asv.anonymization.build_source_reference_pairs \
  --source-manifest data/fisher_manifest.txt \
  --reference-root /path/to/LibriSpeech \
  --out data/source_reference_pairs.txt \
  --metadata-out data/source_reference_pairs.metadata.json \
  --seed 1234 \
  --min-reference-seconds 4.0
```

This samples one reference utterance for every source utterance. References shorter than four seconds are excluded, matching the paper experiments.

## 13. Run Anonymization

```bash
uv run python -m multimodal_asv.anonymization.anonymize_with_streamvoiceanon \
  --pairs-file data/source_reference_pairs.txt \
  --out-root data/anonymized_fisher \
  --config-path configs/config_firefly_arvcasr_8192_delay0_8.yaml \
  --checkpoint-path pretrained_checkpoints/dual_ar_delay_0_8.pth \
  --delay 2 \
  --alpha 1.0 \
  --metadata-out data/anonymized_fisher/metadata.json \
  --mapping-out data/anonymized_fisher/original_to_anonymized.json
```

Update `--pairs-file` and `--out-root` for the run. By default, the configuration and checkpoint paths are resolved relative to the bundled StreamVoiceAnon root. Use absolute paths or pass `--streamvoiceanon-root` when using another installation.

The anonymized waveforms are written as:

```text
<out_root>/<source_speaker>/<source_stem>__to__<reference_speaker>__<reference_stem>.wav
```

For example:

```text
data/anonymized_fisher/fisher_spk1/utt1__to__libri_spk1__ref1.wav
```

Rerun the same command with the same `--out-root` to resume an interrupted run. Existing output files are skipped, and only missing conversions are processed.

## 14. Rewrite Trials and Manifests

Use the generated mapping to replace original audio paths with anonymized paths.

For trial JSONL files:

```bash
uv run python -m multimodal_asv.anonymization.rewrite_to_anonymized_audio \
  --mapping-json data/anonymized_fisher/original_to_anonymized.json \
  --input data/trials_N15.jsonl \
  --output data/trials_N15_anonymized.jsonl \
  --format trial-jsonl \
  --path-resolution absolute
```

For manifests:

```bash
uv run python -m multimodal_asv.anonymization.rewrite_to_anonymized_audio \
  --mapping-json data/anonymized_fisher/original_to_anonymized.json \
  --input data/audio_only_manifest.txt \
  --output data/audio_only_manifest_anonymized.txt \
  --format manifest \
  --path-resolution absolute
```

The rewriting step changes only audio paths. Labels, speaker IDs, transcripts, word timestamps, and additional fields are preserved.

## Parallel Anonymization

Divide the source-reference pair file into deterministic shards and launch one process per shard. The following example starts the first of eight workers:

```bash
uv run python -m multimodal_asv.anonymization.anonymize_with_streamvoiceanon \
  --pairs-file data/source_reference_pairs.txt \
  --out-root data/anonymized_fisher \
  --config-path configs/config_firefly_arvcasr_8192_delay0_8.yaml \
  --checkpoint-path pretrained_checkpoints/dual_ar_delay_0_8.pth \
  --delay 2 \
  --alpha 1.0 \
  --num-shards 8 \
  --shard-id 0
```

Launch the remaining workers with `--shard-id 1` through `--shard-id 7`, keeping `--num-shards 8` fixed.

After all shards finish, write the mapping once:

```bash
uv run python -m multimodal_asv.anonymization.anonymize_with_streamvoiceanon \
  --pairs-file data/source_reference_pairs.txt \
  --out-root data/anonymized_fisher \
  --write-mapping-only \
  --mapping-out data/anonymized_fisher/original_to_anonymized.json
```

When completed outputs are stored under another root, provide that directory with `--skip-existing-root`:

```bash
uv run python -m multimodal_asv.anonymization.anonymize_with_streamvoiceanon \
  --pairs-file data/source_reference_pairs.txt \
  --out-root data/anonymized_fisher_new \
  --skip-existing-root /path/to/previous_outputs \
  --config-path configs/config_firefly_arvcasr_8192_delay0_8.yaml \
  --checkpoint-path pretrained_checkpoints/dual_ar_delay_0_8.pth \
  --delay 2 \
  --alpha 1.0
```
