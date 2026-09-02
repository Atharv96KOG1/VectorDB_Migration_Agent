from __future__ import annotations

import os
import stat

from core.checkpointing.store import (
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
    write_artifact,
)


def test_save_and_load_round_trip(tmp_path):
    state = {"migration_id": "mig-1", "history": [{"state": "INIT"}]}
    save_checkpoint("mig-1", state, checkpoint_dir=tmp_path)
    loaded = load_checkpoint("mig-1", checkpoint_dir=tmp_path)
    assert loaded == state


def test_load_missing_checkpoint_returns_none(tmp_path):
    assert load_checkpoint("does-not-exist", checkpoint_dir=tmp_path) is None


def test_list_checkpoints(tmp_path):
    save_checkpoint("mig-a", {"x": 1}, checkpoint_dir=tmp_path)
    save_checkpoint("mig-b", {"x": 2}, checkpoint_dir=tmp_path)
    assert list_checkpoints(checkpoint_dir=tmp_path) == ["mig-a", "mig-b"]


def test_delete_checkpoint(tmp_path):
    save_checkpoint("mig-c", {"x": 1}, checkpoint_dir=tmp_path)
    delete_checkpoint("mig-c", checkpoint_dir=tmp_path)
    assert load_checkpoint("mig-c", checkpoint_dir=tmp_path) is None


def test_migration_id_with_unsafe_characters_is_sanitized_for_the_filename(tmp_path):
    save_checkpoint("mig/../etc", {"x": 1}, checkpoint_dir=tmp_path)
    assert load_checkpoint("mig/../etc", checkpoint_dir=tmp_path) == {"x": 1}
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert ".." not in files[0].name and "/" not in files[0].name


def test_save_checkpoint_restricts_file_permissions_to_owner_only(tmp_path):
    # checkpoint["request"] can hold a raw credential typed directly into a trigger field
    # (2026-09-01) rather than only an env var name, so the file at rest must not be
    # group/world-readable.
    path = save_checkpoint(
        "mig-secret", {"request": {"source_credential_ref": "pcsk_x"}}, checkpoint_dir=tmp_path
    )
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_write_artifact_creates_parent_dirs(tmp_path):
    target_dir = tmp_path / "nested" / "reports"
    path = write_artifact(target_dir, "report.json", {"ok": True})
    assert (target_dir / "report.json").exists()
    assert path.endswith("report.json")
