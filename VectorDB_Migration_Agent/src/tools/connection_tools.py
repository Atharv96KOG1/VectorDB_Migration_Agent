"""CONNECT state (V2 §7): credential lookup -> authentication check for both source and
target. Assumes INIT (tools/checkpoint_tools.init_checkpoint) already created the
checkpoint record for this migration_id.
"""

from __future__ import annotations

from aetherion_sdk import tool

from core.checkpointing.store import load_checkpoint, save_checkpoint
from tools._shared import build_adapter


@tool()
async def connect_and_validate(migration_id: str) -> dict:
    checkpoint = load_checkpoint(migration_id)
    if checkpoint is None:
        raise RuntimeError(f"no checkpoint found for migration_id={migration_id!r}; run INIT first")
    req = checkpoint["request"]

    result: dict = {}
    for side in ("source", "target"):
        provider = req[f"{side}_provider"]
        adapter = build_adapter(
            provider, req[f"{side}_credential_ref"], req.get(f"{side}_endpoint_ref")
        )
        try:
            connected = await adapter.validate_credentials()
        finally:
            await adapter.close()
        result[side] = {"provider": provider, "connected": connected}

    checkpoint["connection"] = result
    checkpoint["history"].append({"state": "CONNECT", "detail": result})
    save_checkpoint(migration_id, checkpoint)

    connected = result["source"]["connected"] and result["target"]["connected"]
    return {"migration_id": migration_id, "connected": connected, "detail": result}
