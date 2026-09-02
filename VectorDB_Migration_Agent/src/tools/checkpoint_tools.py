"""INIT state: create a fresh checkpoint record, or detect an existing one for
cross-run resume (V2 §38-39). `check_resume` is what lets INIT jump straight to the
recorded state instead of restarting a migration that was merely terminated, not failed.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint


@tool()
async def check_resume(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        return {"migration_id": migration_id, "resumable": False, "last_state": None}
    history = checkpoint.get("history", [])
    last_state = history[-1]["state"] if history else None
    return {"migration_id": migration_id, "resumable": True, "last_state": last_state}


@tool()
async def init_checkpoint(migration_request: dict) -> dict:
    migration_id = migration_request["migration_id"]
    checkpoint = {
        "migration_id": migration_id,
        "request": migration_request,
        "history": [{"state": "INIT", "detail": "migration started"}],
    }
    save_checkpoint(migration_id, checkpoint)
    return {"migration_id": migration_id, "state": "INIT"}
