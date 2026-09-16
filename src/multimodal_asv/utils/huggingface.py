from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def resolve_hf_or_local_path(
    value: str | Path,
    *,
    cache_dir: str | Path | None = None,
    repo_type: str = "model",
) -> Path:
    """Resolve a local path or an hf://repo_owner/repo_name/path/to/file reference."""

    text = str(value)
    if not text.startswith("hf://"):
        return Path(text).expanduser()

    repo_id, filename = parse_hf_uri(text)
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )


def parse_hf_uri(uri: str) -> tuple[str, str]:
    """Parse hf://owner/repo/path/to/file into a Hugging Face repo id and filename."""

    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Expected hf:// URI, got: {uri}")

    parts = [part for part in (parsed.netloc, *parsed.path.split("/")) if part]
    if len(parts) < 3:
        raise ValueError(
            "Hugging Face paths must look like hf://owner/repo/path/to/file, "
            f"got: {uri}"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return repo_id, filename


__all__ = ["parse_hf_uri", "resolve_hf_or_local_path"]
