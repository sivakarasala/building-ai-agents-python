# Lesson 6 — File System Tools

## Overview

The agent has a working loop and an empty tool registry. In this lesson you build the four file tools (`read_file`, `write_file`, `list_files`, `delete_file`) and wire them in.

## Key concepts

- Each tool is a `(executor_function, tool_definition)` pair. The executor takes a `dict[str, Any]` and returns a `str`. The tool definition is OpenAI function-calling format.
- All four file tools live in one file (`src/agent/tools/file.py`) — they're cohesive.
- Errors should be returned as strings, not raised. The model needs to *see* the error to recover.

## Code

### `src/agent/tools/file.py` (NEW)

```python
import os
from typing import Any


def read_file_execute(args: dict[str, Any]) -> str:
    file_path = args["path"]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file_execute(args: dict[str, Any]) -> str:
    file_path = args["path"]
    content = args["content"]
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_files_execute(args: dict[str, Any]) -> str:
    directory = args.get("directory", ".")
    try:
        entries = os.listdir(directory)
        items = []
        for entry in sorted(entries):
            full_path = os.path.join(directory, entry)
            entry_type = "[dir]" if os.path.isdir(full_path) else "[file]"
            items.append(f"{entry_type} {entry}")
        return "\n".join(items) if items else f"Directory {directory} is empty"
    except FileNotFoundError:
        return f"Error: Directory not found: {directory}"
    except Exception as e:
        return f"Error listing directory: {e}"


def delete_file_execute(args: dict[str, Any]) -> str:
    file_path = args["path"]
    try:
        os.unlink(file_path)
        return f"Successfully deleted {file_path}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error deleting file: {e}"


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read"},
            },
            "required": ["path"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List all files and directories in the specified directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "default": "."},
            },
        },
    },
}

DELETE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Delete a file at the specified path. Use with caution — irreversible.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
}
```

### `src/agent/tools/__init__.py` — wire all four into the registry

```python
from src.agent.tools.file import (
    read_file_execute, write_file_execute,
    list_files_execute, delete_file_execute,
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
)

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file_execute,
    "write_file": write_file_execute,
    "list_files": list_files_execute,
    "delete_file": delete_file_execute,
}

ALL_TOOLS = [
    READ_FILE_TOOL, WRITE_FILE_TOOL,
    LIST_FILES_TOOL, DELETE_FILE_TOOL,
]

FILE_TOOLS = ALL_TOOLS[:]
FILE_TOOL_EXECUTORS = dict(TOOL_EXECUTORS)
```

### `evals/data/file_tools.json` — copy from the [done branch](https://github.com/sivakarasala/building-ai-agents-python/blob/done/evals/data/file_tools.json)

### `evals/file_tools_eval.py` — copy from the [done branch](https://github.com/sivakarasala/building-ai-agents-python/blob/done/evals/file_tools_eval.py)

## Exercises

1. Add an `edit_file` tool that takes `path`, `find`, `replace` and rewrites just the matching substring.
2. Make `list_files` recursive with a `max_depth` parameter.
3. Run the `file_tools_eval` and inspect which prompts confuse the model.
