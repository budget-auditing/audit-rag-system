from __future__ import annotations

from pathlib import Path

DEFAULT_AUDIT_SPLITS_DIR = "datasets-public/audit-ociqa-demo/splits"
LEGACY_OUTPUT_SUBDIRS = {"v3", "retrieval", "generation", "eval", "evidence"}


def is_legacy_output_path(path: str) -> bool:
    parts = Path(path).parts
    for idx, part in enumerate(parts[:-1]):
        if part == "runs" and idx + 1 < len(parts):
            if parts[idx + 1] in LEGACY_OUTPUT_SUBDIRS:
                return True
    return False


def validate_non_legacy_output_path(path: str, label: str = "output") -> None:
    if not path:
        return
    if is_legacy_output_path(path):
        allowed_hint = "runs/audit-ociqa/<model>/..., runs/finqa/<model>/..."
        raise ValueError(
            f"{label} must not be written into the flat legacy `runs/` layout: "
            f"{path}. Use a per-dataset/per-model directory instead, e.g. "
            f"{allowed_hint}"
        )
