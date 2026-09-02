"""Regression test for a real bug found live, 2026-09-01: random_projection became an
exact isometry for equal/expanding dimensions (core/transformations/random_projection.py),
so it can now tie direct_copy bit-for-bit on every quality metric in an equal-dimension
migration. The generic multi-objective optimizer (core/optimizer/scorer.py) then let
latency_ms_p95 — a wall-clock measurement dominated by shared ephemeral-collection RPC
overhead common to every candidate, not real per-transform cost — decide the winner,
flipping between direct_copy and random_projection across identical runs of the exact
same data. Fix: direct_copy, once it passes the quality gate at all, wins outright — it
carries zero transform risk by construction (dimension/metric/dtype were already fully
compatible, so no semantic transformation happened), and must never lose to noise.
"""

from __future__ import annotations

from core.checkpointing.store import load_checkpoint, save_checkpoint
from tools import planning_tools

_PLAN = {
    "plan_id": "plan-1",
    "migration_id": "mig-1",
    "candidates": [
        {"strategy": "direct_copy", "status": "possible", "is_stub": False, "reason": "ok"},
        {"strategy": "random_projection", "status": "possible", "is_stub": False, "reason": "ok"},
    ],
    "selected_strategy": None,
    "selected_benchmark_id": None,
}


def _checkpoint(direct_copy_latency: float, random_projection_latency: float) -> dict:
    return {
        "migration_id": "mig-1",
        "request": {"migration_id": "mig-1"},
        "history": [],
        "plan": _PLAN,
        "benchmark_results": [
            {
                "benchmark_id": "mig-1-direct_copy-a",
                "strategy": "direct_copy",
                "recall_at_10": 1.0,
                "ndcg_at_10": 1.0,
                "topk_overlap": 1.0,
                "passed_quality_gate": True,
                "estimated_cost": None,
                "latency_ms_p95": direct_copy_latency,
            },
            {
                "benchmark_id": "mig-1-random_projection-b",
                "strategy": "random_projection",
                "recall_at_10": 1.0,
                "ndcg_at_10": 1.0,
                "topk_overlap": 1.0,
                "passed_quality_gate": True,
                "estimated_cost": None,
                "latency_ms_p95": random_projection_latency,
            },
        ],
    }


def _patch_checkpoint_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        planning_tools, "load_checkpoint", lambda mid: load_checkpoint(mid, checkpoint_dir=tmp_path)
    )
    monkeypatch.setattr(
        planning_tools,
        "save_checkpoint",
        lambda mid, cp: save_checkpoint(mid, cp, checkpoint_dir=tmp_path),
    )


async def test_direct_copy_wins_even_when_a_tied_candidate_has_lower_measured_latency(
    tmp_path, monkeypatch
):
    # random_projection measured "faster" here purely by RPC-timing luck — direct_copy
    # must still win, since letting noisy latency decide a tied-quality contest is exactly
    # the bug this test guards against.
    save_checkpoint(
        "mig-1", _checkpoint(direct_copy_latency=180.0, random_projection_latency=150.0),
        checkpoint_dir=tmp_path,
    )
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    result = await planning_tools.select_strategy("mig-1")
    assert result["selected"] is True
    assert result["strategy"] == "direct_copy"


async def test_direct_copy_wins_even_when_it_has_higher_measured_latency(tmp_path, monkeypatch):
    save_checkpoint(
        "mig-1", _checkpoint(direct_copy_latency=999.0, random_projection_latency=1.0),
        checkpoint_dir=tmp_path,
    )
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    result = await planning_tools.select_strategy("mig-1")
    assert result["strategy"] == "direct_copy"
