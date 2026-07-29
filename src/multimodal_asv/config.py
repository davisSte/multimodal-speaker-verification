from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Model-family selection plus model-specific options."""

    family: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    """Training or evaluation data description."""

    manifest: Path | None = None
    sample_rate: int = 16_000
    segment_seconds: float = 4.0
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    """Common optimization settings."""

    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment config loaded from YAML."""

    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config and normalize its common sections."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    model_raw = _mapping(raw.get("model"), "model")
    family = model_raw.get("family")
    if not isinstance(family, str) or not family:
        raise ValueError("Config requires model.family.")

    data_raw = _optional_mapping(raw.get("data"), "data")
    training_raw = _optional_mapping(raw.get("training"), "training")
    output_raw = raw.get("output_dir")

    return ExperimentConfig(
        name=str(raw.get("name") or config_path.stem),
        model=ModelConfig(
            family=family,
            options=_without_keys(model_raw, {"family"}),
        ),
        data=DataConfig(
            manifest=_optional_path(data_raw.get("manifest")),
            sample_rate=int(data_raw.get("sample_rate", 16_000)),
            segment_seconds=float(data_raw.get("segment_seconds", 4.0)),
            options=_without_keys(data_raw, {"manifest", "sample_rate", "segment_seconds"}),
        ),
        training=TrainingConfig(
            epochs=int(training_raw.get("epochs", 1)),
            batch_size=int(training_raw.get("batch_size", 16)),
            learning_rate=float(training_raw.get("learning_rate", 1e-4)),
            weight_decay=float(training_raw.get("weight_decay", 0.0)),
            grad_clip=float(training_raw.get("grad_clip", 0.0)),
            options=_without_keys(
                training_raw,
                {"epochs", "batch_size", "learning_rate", "weight_decay", "grad_clip"},
            ),
        ),
        output_dir=_optional_path(output_raw),
        raw=raw,
    )


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section {section!r} must be a mapping.")
    return value


def _optional_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, section)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _without_keys(values: Mapping[str, Any], keys: set[str]) -> Mapping[str, Any]:
    return {key: value for key, value in values.items() if key not in keys}
