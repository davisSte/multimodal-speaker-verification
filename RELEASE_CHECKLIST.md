# Release Checklist

This file tracks pieces that are still missing or may need to be added for a fuller public release.

## Data Preparation

- Fisher data split generation used for training, validation, and evaluation
- manifest creation scripts for audio-only and audio-text training
- prosody feature preparation inputs and any transcript-to-timestamp preprocessing
- trial-file generation code
- code to build enroll/test trial JSONL files from source metadata

## Anonymization

- code used to create the anonymized audio data
- code used to create anonymized trial sets
- documentation for which anonymization setting corresponds to each released experiment
- any metadata mapping needed to connect original and anonymized utterances

## StreamVoiceAnon Integration

- decide whether anonymization should live inside `src/multimodal_asv/` or remain an optional external wrapper
- add optional dependency groups for StreamVoiceAnon requirements if integrated into this repo
- document pretrained anonymization checkpoints and where to obtain them
- add adapters from ASV manifests and trial JSONL files to anonymization inputs
- define how reference audio is selected for anonymization runs
- define how anonymized outputs are written back into ASV manifests and trials
- document anonymization hyperparameters used in experiments such as delay, alpha, and number of reference utterances
- add a batch anonymization script that runs over dataset manifests rather than one file pair at a time
- add an evaluation bridge from anonymized outputs into the current ASV evaluation scripts
- review third-party licenses and redistribution constraints for anonymization checkpoints and vendored code

## Training And Evaluation Reproducibility

- exact commands used for each paper model family
- expected checkpoint naming and selection policy for evaluation
- scoring recipes used to produce paper numbers from saved checkpoints
- description of how text-only LUAR evaluation trials were prepared
- any seed or run-averaging policy used in final reporting

## External Dependencies And Assets

- instructions for obtaining LUAR checkpoints used in fine-tuned runs
- instructions for obtaining any prosody resources required beyond transcripts
- confirmation of whether Whisper model downloads are sufficient or whether a pinned local artifact was used
- note on any Hugging Face authentication requirements for text models

## Results And Paper Alignment

- mapping from each paper result row to the corresponding config and evaluation command
- statement of which tables in the paper are fully reproducible from this repo
- optional script to summarize scores into the paper metrics format

## Examples And Validation

- small example manifests and trial files that can run end-to-end without private data
- smoke-test recipe for training dry-run and evaluation dry-run
- expected output structure for training runs and evaluation outputs

## Documentation

- data preparation walkthrough once the scripts are added
- anonymization workflow documentation
- model-by-model reproduction notes if any model needs special handling
- dataset assumptions, licenses, and access restrictions for non-public source data
