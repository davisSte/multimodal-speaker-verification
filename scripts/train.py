#!/usr/bin/env python3
"""Train a released multimodal ASV model from a paper-facing YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_asv.config import load_config
from multimodal_asv.data import build_data_pipeline
from multimodal_asv.registry import get_model_family
from multimodal_asv.registry import build_model
from multimodal_asv.training import Trainer
from multimodal_asv.utils.assets import resolve_output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and dry-run validation."""
    parser = argparse.ArgumentParser(description="Train a multimodal ASV model from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the selected model family without training.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the configured model, optionally validate data loading, and train."""
    args = parse_args()
    config = load_config(args.config)
    family = get_model_family(config.model.family)
    model = build_model(config)

    print(f"Experiment: {config.name}")
    print(f"Model: {family.display_name} ({family.key})")
    print(f"Category: {family.category}")
    print(f"Model class: {type(model).__name__}")
    if config.data.manifest is not None:
        print(f"Manifest: {config.data.manifest}")
    print(f"Output directory: {resolve_output_dir(config.output_dir, experiment_name=config.name)}")

    if args.dry_run:
        if config.data.manifest is not None and config.data.manifest.exists():
            data = build_data_pipeline(config)
            print(f"Dataset class: {type(data.dataset).__name__}")
            print(f"Dataset size: {len(data.dataset)}")
            print(f"Batches per epoch: {len(data.dataloader)}")
        elif config.data.manifest is not None:
            print("Dataset check skipped: manifest does not exist.")
        print("Dry run complete: config is valid.")
        return

    data = build_data_pipeline(config)
    print(f"Dataset class: {type(data.dataset).__name__}")
    print(f"Dataset size: {len(data.dataset)}")
    print(f"Batches per epoch: {len(data.dataloader)}")
    trainer = Trainer(config)
    history = trainer.fit(model, data.dataloader)
    final = history[-1]
    print(
        f"Training complete: epoch={int(final['epoch'])} "
        f"loss={final['loss']:.4f} acc1={final['acc1']:.2f}",
    )


if __name__ == "__main__":
    main()
