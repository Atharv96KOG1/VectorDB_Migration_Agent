"""Aetherion SDK project for building AI agents and tools."""

import os
import sys

# core/ and tools/ import each other (and get imported by agent/agent.py and every
# tools/*.py module) as bare top-level packages ("core.models...", "tools._shared...")
# rather than "src.core...", matching a conventional src-layout. Whichever way the
# Aetherion worker discovers/imports this package's modules, this package (src/) itself
# is guaranteed to be imported first — so this is the one place that's certain to run
# before anything needs the bare imports to resolve. Not runtime-verified against a live
# Aetherion worker (none available in this sandbox); see docs/ARCHITECTURE.md.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from .agent.agent import VectorDB_Migration_Agent  # noqa: E402

__all__ = ["VectorDB_Migration_Agent"]
