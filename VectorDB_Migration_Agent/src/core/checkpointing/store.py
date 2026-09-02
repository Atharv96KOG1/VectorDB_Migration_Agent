"""JSON checkpoint store (V2 §38). Scoped to exactly the two things Temporal's own
durable workflow history doesn't already give for free: (1) cross-run resume — an
operator terminates a workflow execution, and a *new* execution later needs to pick up
mid-migration, which Temporal has no built-in primitive for; (2) a human-readable audit
artifact independent of Temporal's bounded history-retention window. Checkpoint writes
happen only inside tools (activities), never in the workflow body — see
src/agent/agent.py and tests/unit/test_agent_import_boundary.py.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class CheckpointSettings(BaseSettings):
    """Reads MIGRATION_CHECKPOINT_DIR from the environment, once, at import time — never
    hardcoded, never a raw secret (there's nothing secret here anyway, just a filesystem
    path). Safe to read here: core/checkpointing is never imported by src/agent/agent.py
    (see tests/unit/test_agent_import_boundary.py's FORBIDDEN set, which doesn't include
    it), so this doesn't cross the workflow-determinism boundary tools/_shared.py's
    docstring describes for os.environ access."""

    model_config = SettingsConfigDict(env_prefix="MIGRATION_")

    checkpoint_dir: Path = Path(".migration_checkpoints")


DEFAULT_CHECKPOINT_DIR = CheckpointSettings().checkpoint_dir


def _safe_id(migration_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in migration_id)


def _path_for(migration_id: str, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{_safe_id(migration_id)}.json"


def save_checkpoint(
    migration_id: str, state: dict, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
) -> str:
    # checkpoint["request"] can now hold a raw credential typed directly into a trigger
    # field (tools/_shared.py, 2026-09-01) rather than only an env var name — restrict to
    # owner-read/write so a raw secret at rest isn't world/group-readable on disk.
    path = _path_for(migration_id, checkpoint_dir)
    path.write_text(json.dumps(state, indent=2, default=str))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return str(path)


def load_checkpoint(
    migration_id: str, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
) -> dict | None:
    path = _path_for(migration_id, checkpoint_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_checkpoints(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> list[str]:
    if not checkpoint_dir.exists():
        return []
    return sorted(p.stem for p in checkpoint_dir.glob("*.json"))


def delete_checkpoint(migration_id: str, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> None:
    path = _path_for(migration_id, checkpoint_dir)
    if path.exists():
        path.unlink()


def write_artifact(directory: Path, filename: str, data: dict) -> str:
    """Generic durable-JSON-artifact writer, reused for provenance records and audit
    reports (tools/report_tools.py) so there's exactly one place that owns
    "write JSON to disk, create parent dirs, return the path"."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)
