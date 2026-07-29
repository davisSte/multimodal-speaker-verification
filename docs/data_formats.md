# Data Formats

## Audio-only Manifest

One utterance per line:

```text
speaker_id /path/to/audio.wav
speaker_id /path/to/another_audio.wav
```

## Audio-text Manifest

One utterance plus transcript per line:

```text
speaker_id /path/to/audio.wav transcript text goes here
speaker_id /path/to/another_audio.wav another transcript
```

For hybrid audio-text training, everything after the audio path is treated as transcript text.

## Text-only Trial JSONL

Each JSONL row should contain:

- `label`
- `enroll.transcripts`
- `test.transcripts`

Example:

```json
{"label": 1, "enroll": {"transcripts": ["utt one", "utt two"]}, "test": {"transcripts": ["utt three"]}}
```

## Audio Or Multimodal Trial JSONL

Each row should contain:

- `label`
- `enroll.audio_paths`
- `test.audio_paths`
- optional `enroll.transcripts`
- optional `test.transcripts`

Example:

```json
{"label": 1, "enroll": {"audio_paths": ["audio/spk1_1.wav", "audio/spk1_2.wav"], "transcripts": ["utt one", "utt two"]}, "test": {"audio_paths": ["audio/spk1_3.wav"], "transcripts": ["utt three"]}}
```
