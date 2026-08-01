"""Batch wrapper for StreamVoiceAnon source/reference anonymization."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceReferencePair:
    """One StreamVoiceAnon conversion request."""

    source_wav: str
    source_speaker: str
    reference_wav: str
    reference_speaker: str


@dataclass
class AnonymizationResult:
    """Counters collected while processing a pair file."""

    assigned: int = 0
    converted: int = 0
    skipped_existing: int = 0
    malformed: int = 0
    missing_inputs: int = 0
    failed: int = 0


def resolve_under_root(path: str | Path, root: Path) -> Path:
    """Resolve a path under a root unless it is already absolute."""

    value = Path(path)
    return value if value.is_absolute() else root / value


def resolve_pair_audio_path(path: str | Path, pairs_file: Path) -> Path:
    """Resolve pair-file audio paths relative to the pair file when needed."""

    value = Path(path).expanduser()
    return value if value.is_absolute() else (pairs_file.parent / value).resolve()


@contextmanager
def working_directory(path: Path):
    """Temporarily run code from a directory."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def safe_stem(path: str | Path) -> str:
    """Return a filename stem that is safe to use in deterministic outputs."""

    return Path(path).stem.replace(" ", "_")


def parse_pair_line(line: str) -> SourceReferencePair | None:
    """Parse one source/reference pair line.

    Expected format:

        source_wav|source_speaker|reference_wav|reference_speaker
    """

    raw = line.strip()
    if not raw:
        return None

    parts = raw.split("|")
    if len(parts) != 4:
        return None

    return SourceReferencePair(
        source_wav=parts[0],
        source_speaker=parts[1],
        reference_wav=parts[2],
        reference_speaker=parts[3],
    )


def output_relative_path(pair: SourceReferencePair) -> Path:
    """Return the deterministic output path relative to the output root."""

    source_stem = safe_stem(pair.source_wav)
    reference_stem = safe_stem(pair.reference_wav)
    return Path(pair.source_speaker) / f"{source_stem}__to__{pair.reference_speaker}__{reference_stem}.wav"


def build_original_to_anonymized_mapping(
    pairs_file: str | Path,
    out_root: str | Path,
) -> dict[str, str]:
    """Build an original-audio to anonymized-audio path mapping from a pair file."""

    pairs_path = Path(pairs_file).expanduser().resolve()
    output_root = Path(out_root).expanduser().resolve()
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")

    mapping: dict[str, str] = {}
    with pairs_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_num, line in enumerate(handle, start=1):
            pair = parse_pair_line(line)
            if pair is None:
                continue
            source_path = resolve_pair_audio_path(pair.source_wav, pairs_path)
            anonymized_path = output_root / output_relative_path(pair)
            previous = mapping.get(str(source_path))
            if previous is not None and previous != str(anonymized_path):
                raise ValueError(f"Source path maps to multiple anonymized files at line {line_num}: {source_path}")
            mapping[str(source_path)] = str(anonymized_path)
    return mapping


def write_original_to_anonymized_mapping(
    mapping_out: str | Path,
    pairs_file: str | Path,
    out_root: str | Path,
) -> dict[str, str]:
    """Write an original-to-anonymized JSON mapping and return it."""

    mapping = build_original_to_anonymized_mapping(pairs_file, out_root)
    mapping_path = Path(mapping_out).expanduser().resolve()
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return mapping


class StreamVoiceAnonBatchRunner:
    """Run StreamVoiceAnon inference over source/reference pair files."""

    def __init__(
        self,
        *,
        streamvoiceanon_root: str | Path,
        config_path: str | Path = "configs/config_firefly_arvcasr_8192_delay0_8.yaml",
        checkpoint_path: str | Path = "pretrained_checkpoints/dual_ar_delay_0_8.pth",
        compile_ar: bool = False,
    ) -> None:
        self.streamvoiceanon_root = Path(streamvoiceanon_root).expanduser().resolve()
        self.config_path = resolve_under_root(config_path, self.streamvoiceanon_root)
        self.checkpoint_path = resolve_under_root(checkpoint_path, self.streamvoiceanon_root)
        self.compile_ar = compile_ar

        self._validate_paths()
        self._model = self._load_model()

    def _validate_paths(self) -> None:
        if not self.streamvoiceanon_root.is_dir():
            raise FileNotFoundError(f"StreamVoiceAnon root not found: {self.streamvoiceanon_root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"StreamVoiceAnon config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"StreamVoiceAnon checkpoint not found: {self.checkpoint_path}")

    def _load_model(self):
        sys.path.insert(0, str(self.streamvoiceanon_root))
        from evaluations.infer_arvc import InferenceWrapper  # pylint: disable=import-error,import-outside-toplevel

        with working_directory(self.streamvoiceanon_root):
            return InferenceWrapper(
                str(self.config_path),
                str(self.checkpoint_path),
                compile_ar=self.compile_ar,
                compile_decoder=False,
                compile_encoder=False,
            )

    def run_pair_file(
        self,
        *,
        pairs_file: str | Path,
        out_root: str | Path,
        delay: int = 2,
        alpha: float = 1.0,
        num_shards: int = 1,
        shard_id: int = 0,
        skip_existing_roots: Iterable[str | Path] = (),
        log_every: int = 100,
    ) -> AnonymizationResult:
        """Run anonymization for one shard of a source/reference pair file."""

        if num_shards <= 0:
            raise ValueError("num_shards must be > 0")
        if not 0 <= shard_id < num_shards:
            raise ValueError("shard_id must be in [0, num_shards)")

        pairs_path = Path(pairs_file).expanduser().resolve()
        output_root = Path(out_root).expanduser().resolve()
        existing_roots = [Path(path).expanduser().resolve() for path in skip_existing_roots]

        if not pairs_path.is_file():
            raise FileNotFoundError(f"Pair file not found: {pairs_path}")
        for root in existing_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Skip-existing root not found: {root}")

        result = AnonymizationResult()
        output_root.mkdir(parents=True, exist_ok=True)

        with pairs_path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file):
                if line_index % num_shards != shard_id:
                    continue

                result.assigned += 1
                pair = parse_pair_line(line)
                if pair is None:
                    result.malformed += 1
                    continue

                source_path = resolve_pair_audio_path(pair.source_wav, pairs_path)
                reference_path = resolve_pair_audio_path(pair.reference_wav, pairs_path)
                if not source_path.is_file() or not reference_path.is_file():
                    result.missing_inputs += 1
                    continue

                relative_out = output_relative_path(pair)
                output_path = output_root / relative_out
                probe_paths = [output_path] + [root / relative_out for root in existing_roots]
                if any(path.is_file() for path in probe_paths):
                    result.skipped_existing += 1
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with working_directory(self.streamvoiceanon_root):
                        self._model.infer(
                            src_path=str(source_path),
                            ref_path=str(reference_path),
                            output_path=str(output_path),
                            delay=delay,
                            alpha=alpha,
                            save_result=True,
                        )
                    result.converted += 1
                except Exception as exc:  # noqa: BLE001
                    result.failed += 1
                    logging.exception(
                        "Conversion failed for source=%s reference=%s: %s",
                        pair.source_wav,
                        pair.reference_wav,
                        exc,
                    )

                if log_every > 0 and result.assigned % log_every == 0:
                    logging.info("Progress shard %d/%d: %s", shard_id, num_shards, asdict(result))

        logging.info("Finished shard %d/%d: %s", shard_id, num_shards, asdict(result))
        return result


def write_metadata(
    *,
    metadata_out: str | Path,
    pairs_file: str | Path,
    out_root: str | Path,
    runner: StreamVoiceAnonBatchRunner,
    delay: int,
    alpha: float,
    num_shards: int,
    shard_id: int,
    skip_existing_roots: Iterable[str | Path],
) -> None:
    """Write reproducibility metadata for a StreamVoiceAnon batch run."""

    metadata_path = Path(metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pairs_file": str(Path(pairs_file)),
        "out_root": str(Path(out_root)),
        "streamvoiceanon_root": str(runner.streamvoiceanon_root),
        "config_path": str(runner.config_path),
        "checkpoint_path": str(runner.checkpoint_path),
        "delay": delay,
        "alpha": alpha,
        "num_shards": num_shards,
        "shard_id": shard_id,
        "compile": runner.compile_ar,
        "skip_existing_root": [str(Path(path)) for path in skip_existing_roots],
        "pair_format": "source_wav|source_speaker|reference_wav|reference_speaker",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
