"""Asserts src/agent/agent.py's import graph never pulls in numpy/scipy/sklearn/
qdrant_client/httpx — Temporal's workflow sandbox constrains the *module* import graph,
not just the function body, so a transitive import here could break workflow loading or
cause confusing sandbox behavior even if the workflow body itself never touches numpy.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN = {
    "numpy",
    "scipy",
    "sklearn",
    "qdrant_client",
    "httpx",
    "core.adapters",
    "core.transformations",
    "core.benchmarking",
}

AGENT_FILE = Path(__file__).resolve().parents[2] / "src" / "agent" / "agent.py"


def _imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_agent_module_has_no_forbidden_direct_imports():
    source = AGENT_FILE.read_text()
    imported = _imported_names(source)
    for forbidden in FORBIDDEN:
        assert not any(
            name == forbidden or name.startswith(forbidden + ".") for name in imported
        ), f"src/agent/agent.py must not directly import {forbidden!r}: found in {imported}"


def test_agent_module_transitive_import_graph_is_clean():
    """Imports ONLY agent.agent in a fresh subprocess and inspects sys.modules for
    anything under the forbidden set — catches transitive pulls a static AST scan of
    agent.py alone would miss (e.g. via core.models.workflow_state). Must be a fresh
    process: within the same pytest session, other test modules already import
    numpy/etc. first, which would hide a real violation (or produce a false positive)
    depending on collection order.
    """
    src_dir = str(AGENT_FILE.parents[1])
    code = (
        f"import sys; sys.path.insert(0, {src_dir!r})\n"
        "import agent.agent\n"
        f"forbidden = {sorted(FORBIDDEN)!r}\n"
        "offenders = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden))\n"
        "print(','.join(offenders))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    offenders = [m for m in result.stdout.strip().split(",") if m]
    assert (
        not offenders
    ), f"importing agent.agent transitively loaded forbidden module(s): {offenders}"
