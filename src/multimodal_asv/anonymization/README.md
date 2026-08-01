# Anonymization Package

This package contains the StreamVoiceAnon integration used by the repository.

- `streamvoiceanon.py` provides the reusable Python wrapper for batch anonymization from source-reference pair files.
- `StreamVoiceAnon/` contains the vendored StreamVoiceAnon source and configs used by the wrapper.
- `anonymize_with_streamvoiceanon.py` is the command-line module documented in the repository README.

The StreamVoiceAnon pretrained checkpoints are not included. Place downloaded checkpoints under:

```text
src/multimodal_asv/anonymization/StreamVoiceAnon/pretrained_checkpoints/
```

or pass an absolute checkpoint path to the command-line script.
