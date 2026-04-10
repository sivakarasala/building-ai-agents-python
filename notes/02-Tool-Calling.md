# Lesson 2 — Tool Calling

## Overview

A model alone can only emit text. To do anything useful — read a file, run a query, hit an API — it needs *tools*. In this lesson you set up the two halves of tool calling: a **registry** that describes the tools to the model in OpenAI's function-calling format, and an **executor** that dispatches a tool name + arguments to the actual Python function. We don't add any real tools yet (those come in lesson 6); we build the plumbing so dropping tools in later is one line.

## Key concepts

- **Two-sided definition.** Each tool has (1) a JSON schema the model sees (`ALL_TOOLS`), and (2) a Python callable that runs when the model picks it (`TOOL_EXECUTORS`). Keeping these in lockstep is the registry's whole job.
- **OpenAI function format.** A tool definition is `{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...JSON schema...}}}`. The model uses the description + parameter docs to decide *when* to call it.
- **Dispatcher.** `execute_tool(name, args)` is one tiny function that looks up the executor in the registry, calls it, and stringifies the result. Errors and unknown tools are turned into strings the model can read on the next turn.
- **Why separate?** The agent loop (next lesson) calls `execute_tool` without knowing which tools exist. Adding a new tool means appending to the registry — no edits to the loop.

## Code

### `src/agent/tools/__init__.py`

```python
# No tools yet — file tools will be added in lesson 6.

TOOL_EXECUTORS: dict[str, callable] = {}

ALL_TOOLS: list[dict] = []
```

### `src/agent/execute_tool.py`

```python
from typing import Any
from src.agent.tools import TOOL_EXECUTORS


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments."""
    executor = TOOL_EXECUTORS.get(name)

    if executor is None:
        return f"Unknown tool: {name}"

    try:
        result = executor(args)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {e}"
```

### `tests/test_execute_tool.py`

```python
from src.agent.execute_tool import execute_tool
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS


def test_execute_tool_unknown():
    out = execute_tool("nope", {})
    assert "Unknown tool" in out


def test_registry_starts_empty():
    # File tools will be added in lesson 6.
    assert TOOL_EXECUTORS == {}
    assert ALL_TOOLS == []
```

## Exercises

1. **Add a fake tool.** Register a `get_time` tool with a Python `lambda args: str(datetime.now())` and confirm `execute_tool("get_time", {})` works.
2. **Schema sketch.** Write the JSON schema for a `read_file` tool with one required string parameter `path`. Don't implement it yet — that's lesson 6.
3. **Error path.** Register a tool whose executor raises `ValueError` and verify `execute_tool` returns a string starting with `"Error executing"`.
