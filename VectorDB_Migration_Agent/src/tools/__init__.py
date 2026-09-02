"""Tools package. Importing this package registers every `@tool()` activity with
aetherion_sdk's TOOL_REGISTRY — belt-and-suspenders alongside whatever filesystem-scan
discovery the Aetherion tool worker may also do, so registration doesn't silently depend
on which mechanism actually runs.
"""

from . import (
    _shared,  # noqa: F401
    _transform_factory,  # noqa: F401
    benchmark_tools,  # noqa: F401
    checkpoint_tools,  # noqa: F401
    compatibility_tools,  # noqa: F401
    connection_tools,  # noqa: F401
    discovery_tools,  # noqa: F401
    execution_tools,  # noqa: F401
    planning_tools,  # noqa: F401
    report_tools,  # noqa: F401
    transform_tools,  # noqa: F401
    validation_tools,  # noqa: F401
)
