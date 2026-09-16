from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def resolve_asset_path(
    asset_path: str | Path | None,
    *,
    allow_download: bool = False,
    download_url: str | None = None,
    download_root: str | Path | None = None,
    default_filename: str,
    asset_name: str,
) -> Path | None:
    """Resolve a local asset path, optionally downloading it into a user-owned directory."""

    candidate = Path(str(asset_path)).expanduser() if asset_path is not None else None
    if candidate is not None and candidate.exists():
        return candidate
    target_root = Path(download_root).expanduser() if download_root is not None else None
    if candidate is not None and not candidate.is_absolute() and target_root is not None:
        rooted_candidate = target_root / candidate.name
        if rooted_candidate.exists():
            return rooted_candidate

    url = download_url
    if url is None and asset_path is not None:
        asset_path_str = str(asset_path)
        parsed = urlparse(asset_path_str)
        if parsed.scheme in {"http", "https"}:
            url = asset_path_str

    if url is None:
        return candidate
    if not allow_download:
        raise FileNotFoundError(
            f"{asset_name} was not found locally at {candidate}, and downloading is disabled. "
            f"Set allow_download=true and provide a writable download_root."
        )

    target_root = Path(download_root or Path.home() / ".cache" / "multimodal_asv" / "assets").expanduser()
    target_root.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or default_filename
    target_path = target_root / filename
    if not target_path.exists():
        logging.info("Downloading %s from %s to %s", asset_name, url, target_path)
        from torch.hub import download_url_to_file

        download_url_to_file(url, str(target_path), progress=True)
    return target_path


def with_default_download_root(
    encoder_config: dict[str, Any] | None,
    *,
    output_dir: str | Path | None,
    experiment_name: str,
) -> dict[str, Any] | None:
    """Fill encoder download_root from output_dir/assets when unset."""

    if encoder_config is None:
        return None
    resolved = dict(encoder_config)
    if resolved.get("download_root") is None:
        resolved_output_dir = resolve_output_dir(output_dir, experiment_name=experiment_name)
        resolved["download_root"] = str(resolved_output_dir / "assets")
    return resolved


def resolve_output_dir(output_dir: str | Path | None, *, experiment_name: str) -> Path:
    """Resolve the effective output directory."""

    if output_dir is not None:
        return Path(output_dir).expanduser()
    return Path("runs") / experiment_name


__all__ = ["resolve_asset_path", "resolve_output_dir", "with_default_download_root"]
