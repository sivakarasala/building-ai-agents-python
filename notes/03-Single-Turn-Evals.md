# Lesson 3 — Single-turn Evals

## Overview

Now that the model can call tools, how do you know it picks the *right* tool for a given prompt? You write evals. Single-turn evals don't run the full agent loop — they just ask the model "given this prompt and these tools, which tool(s) would you call?" and score the answer. They are cheap, fast, and great for catching regressions when you change the system prompt or tool descriptions.

## Key concepts

- **Eval data** — a small JSON-ish record with a `prompt`, the `tools` available to the model, and a `target` describing which tools were expected (or forbidden).
- **Categories** — `golden` (must pick the right tool), `secondary` (nice to have), `negative` (must NOT call any tool).
- **Executor** — runs one non-streaming chat completion with `tool_choice="auto"`, parses the tool calls, and returns the names. No tools are actually executed.
- **Evaluators** — pure functions that take the executor result + target and return a 0..1 score: `tools_selected`, `tools_avoided`, `tool_selection_score` (precision/recall F1).
- **Why no execution?** Single-turn evals are about *selection*, not behavior. Skipping execution makes them deterministic, fast, and free of side effects.

## Code

### `evals/types.py`

```python
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalData:
    """Input data for single-turn tool selection evaluations."""
    prompt: str
    tools: list[str]
    system_prompt: Optional[str] = None
    config: Optional[dict[str, Any]] = None


@dataclass
class EvalTarget:
    """Target expectations for single-turn evaluations."""
    category: str  # "golden", "secondary", or "negative"
    expected_tools: Optional[list[str]] = None
    forbidden_tools: Optional[list[str]] = None


@dataclass
class SingleTurnResult:
    """Result from single-turn executor."""
    tool_calls: list[dict[str, Any]]
    tool_names: list[str]
    selected_any: bool
```

### `evals/utils.py`

```python
from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data."""
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]
```

### `evals/executors.py`

```python
import json
from typing import Any
from openai import OpenAI
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from evals.types import EvalData, SingleTurnResult
from evals.utils import build_messages

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def single_turn_executor(
    data: dict[str, Any],
    available_tools: list[dict],
) -> SingleTurnResult:
    """Run a single-turn evaluation. Gets tool selection without executing."""
    messages = build_messages(data)

    tool_names_wanted = set(data["tools"])
    tools = [
        t for t in available_tools
        if t["function"]["name"] in tool_names_wanted
    ]

    model = "gpt-5-mini"
    if data.get("config") and data["config"].get("model"):
        model = data["config"]["model"]

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        tools=tools if tools else None,
    )

    message = response.choices[0].message

    tool_calls = []
    tool_names = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_calls.append({"tool_name": tc.function.name, "args": args})
            tool_names.append(tc.function.name)

    return SingleTurnResult(
        tool_calls=tool_calls,
        tool_names=tool_names,
        selected_any=len(tool_names) > 0,
    )
```

### `evals/evaluators.py`

```python
from typing import Union
from evals.types import EvalTarget, SingleTurnResult


def tools_selected(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Check if all expected tools were selected. Returns 1 or 0."""
    expected = target.expected_tools
    if not expected:
        return 1.0

    selected = set(output.tool_names)
    return 1.0 if all(t in selected for t in expected) else 0.0


def tools_avoided(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Check if forbidden tools were avoided. Returns 1 or 0."""
    forbidden = target.forbidden_tools
    if not forbidden:
        return 1.0

    selected = set(output.tool_names)
    return 0.0 if any(t in selected for t in forbidden) else 1.0


def tool_selection_score(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Precision/recall F1 score for tool selection. Returns 0 to 1."""
    if not target.expected_tools:
        return 0.5 if output.selected_any else 1.0

    expected = set(target.expected_tools)
    selected = set(output.tool_names)

    hits = len([t for t in output.tool_names if t in expected])
    precision = hits / len(selected) if selected else 0.0
    recall = hits / len(expected) if expected else 0.0

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
```

### `tests/test_evals_utils.py`

```python
from evals.utils import build_messages


def test_build_messages_uses_system_prompt():
    msgs = build_messages({"prompt": "hi"})
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_messages_custom_system_prompt():
    msgs = build_messages({"prompt": "hi", "system_prompt": "you are X"})
    assert msgs[0]["content"] == "you are X"
```

## Exercises

1. **Write 3 golden cases** for the tools you have today (e.g. "delete X" → `delete_file`).
2. **Write 3 negative cases** where the model should refuse to call any tool (e.g. "what's 2+2?").
3. **Break the system prompt** on purpose and re-run the evals — watch your scores drop. Then fix it.
4. **Add a `tool_args_match` evaluator** that checks the model passed the right argument values, not just the right tool name.
