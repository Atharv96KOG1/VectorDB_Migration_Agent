# VectorDB_Migration_Agent

A provider-neutral, benchmark-gated vector database migration agent (currently Pinecone
<-> Qdrant). It treats migration as a **retrieval-preservation problem**: it discovers
capabilities with an honest tri-state (`TRUE`/`FALSE`/`UNKNOWN`), classifies
compatibility, generates and benchmarks transformation candidates (direct copy, PCA,
random projection, MRL truncation, re-embedding — plus honest extension-point stubs for
Matryoshka-Adaptor/Learned Projection/Retrieval-Aware Projection/Knowledge Distillation/
vec2vec), and only selects a strategy that clears a configured Recall@10/NDCG@10 quality
gate, gated on an actually-executed benchmark row (never a hallucinated recommendation).

See `docs/ARCHITECTURE.md` for the full design (state machine, what's real vs. stub and
why, deferred scope). Built as an Aetherion SDK agent (`src/agent/agent.py` orchestrates
a Temporal workflow; `src/tools/*.py` are the activities that do the real work) on top of
a provider-neutral core (`src/core/`).

Run the test suite (no external credentials needed — the integration tests use a local,
on-disk Qdrant instance for both "source" and "target"):

```bash
uv run pytest
```

### Creating a New Project

Use the `aetherion` CLI to scaffold a new project.

```bash

aetherion init VectorDB_Migration_Agent
cd VectorDB_Migration_Agent

uv sync

source .venv/bin/activate
```

## Configuration

```bash
aetherion config init
```

## Write Tools and Agents

Example `agent/metadata.json`:

```json
{
  "name": "VectorDB_Migration_Agent",
  "version": "1.0.0",
  "description": "Agent workflows for My Package.",
  "config": {
    "triggers": [
      {
        "name": "files",
        "type": "file",
        "required": true,
        "description": "Files to be uploaded for processing.",
        "friendly_name": "Upload Files"
      }
    ]
  }
}
```

Supported trigger fields:

- text / str  
- dropdown  
- file / form  
- boolean  
- number  
- textarea  
- default  


### Tool Execution Options

```python
from datetime import timedelta
from aetherion_sdk import toolExecutor

result = await toolExecutor.execute(
    "analyze_text",
    "Some text",
    start_to_close_timeout=timedelta(seconds=5),
)
```

## Running Workers

```bash
aetherion run --tool
```

```bash
aetherion run --agent
```

Trigger an agent:

```bash
aetherion agent VectorDB_Migration_Agent '{"input": "World"}'
```

## Publish

```bash
aetherion publish
```

