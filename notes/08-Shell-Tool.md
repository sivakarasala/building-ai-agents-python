# Lesson 8 — Shell + Code Execution Tools

## Overview

Add two tools that let the agent run arbitrary code on your machine: `run_command` (a shell pass-through) and `execute_code` (write a temp file, run it, return output). These are powerful and dangerous — this lesson is also a setup for Lesson 9 (HITL) which adds the approval gate around them.

## Key concepts

- Both tools use `subprocess.run` with `timeout=30` and `capture_output=True` so they fail safely.
- `execute_code` writes the snippet to a temp file with the right extension, runs it via the language interpreter, and cleans up.
- This is the point in the course where the security tradeoffs of letting an LLM execute code on your box become real. Talk about it.

## Code

### `src/agent/tools/shell.py` (NEW)

```python
import subprocess
from typing import Any


def run_command_execute(args: dict[str, Any]) -> str:
    """Execute a shell command and return its output."""
    command = args["command"]
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr

        if result.returncode != 0:
            return f"Command failed (exit code {result.returncode}):\n{output}"

        return output or "Command completed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Execute a shell command and return its output. Use this for system operations, running scripts, or interacting with the operating system.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
}
```

### `src/agent/tools/code_execution.py` (NEW)

```python
import os
import tempfile
import subprocess
from typing import Any


def execute_code_execute(args: dict[str, Any]) -> str:
    """Execute code by writing to a temp file and running it."""
    code = args["code"]
    language = args.get("language", "python")

    extensions = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
    }

    commands = {
        "python": lambda f: f"python3 {f}",
        "javascript": lambda f: f"node {f}",
        "typescript": lambda f: f"npx tsx {f}",
    }

    ext = extensions.get(language, ".py")
    get_command = commands.get(language)

    if not get_command:
        return f"Unsupported language: {language}"

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_file = f.name

        command = get_command(tmp_file)
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr

        if result.returncode != 0:
            return f"Execution failed (exit code {result.returncode}):\n{output}"

        return output or "Code executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 30 seconds"
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass


EXECUTE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "Execute code for anything you need compute for. Supports Python, JavaScript, and TypeScript. Returns the output of the execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript"],
                    "description": "The programming language of the code",
                    "default": "python",
                },
            },
            "required": ["code"],
        },
    },
}
```

### `src/agent/tools/__init__.py` (MODIFY — wire the new tools into the registry)

Add the imports:

```python
from src.agent.tools.shell import run_command_execute, RUN_COMMAND_TOOL
from src.agent.tools.code_execution import execute_code_execute, EXECUTE_CODE_TOOL
```

Add to `TOOL_EXECUTORS`:

```python
    "run_command": run_command_execute,
    "execute_code": execute_code_execute,
```

Add to `ALL_TOOLS`:

```python
    RUN_COMMAND_TOOL,
    EXECUTE_CODE_TOOL,
```

Add the shell subset:

```python
SHELL_TOOLS = [RUN_COMMAND_TOOL]
SHELL_TOOL_EXECUTORS = {
    "run_command": run_command_execute,
}
```

### `evals/data/shell_tools.json` (NEW)

Copy from the [done branch](https://github.com/sivakarasala/building-ai-agents-python/blob/done/evals/data/shell_tools.json) — 9 cases covering golden / secondary / negative for `run_command`.

### `evals/shell_tools_eval.py` (NEW)

Same shape as `evals/file_tools_eval.py` but loads `shell_tools.json` and uses `SHELL_TOOLS`.

## Exercises

1. Try `"calculate the sum of squares from 1 to 1000"` — does the model pick `execute_code` over running it inline?
2. Try `"what's my disk usage"` — does it correctly pick `run_command` with `df -h`?
3. Add a `working_directory` parameter to `run_command` so the agent can `cd` semantically without leaking shell state.
