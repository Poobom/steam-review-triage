"""SHA1-based artifact cache utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path

ARTIFACTS_ROOT = Path("artifacts")
HASH_PREFIX_LEN = 12


def compute_sha1(file_path: Path) -> str:
    h = hashlib.sha1()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_artifact_dir(input_path: Path, root: Path = ARTIFACTS_ROOT) -> Path:
    digest = compute_sha1(input_path)[:HASH_PREFIX_LEN]
    return root / digest


def stage_done(out_dir: Path, stage_filename: str) -> bool:
    return (out_dir / stage_filename).exists()
