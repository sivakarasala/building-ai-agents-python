# Lesson 5 — Multi-turn Evals + LLM-as-Judge

## Overview

Single-turn evals only test "did the model pick the right tool?". Multi-turn evals test the *agent* — the full loop with tool execution, multiple back-and-forth turns, and a final response. To do this without burning real API calls or touching the file system, you mock the tools. To grade the final response without writing per-test assertions, you use an LLM as the judge.

## Key concepts

- **Mocked tools** — students reuse the function-calling tool definitions but swap the executor for one that returns a hardcoded value. The model still does real tool calling; only the side effects are fake.
- **Multi-turn executor** — runs the agent loop with the mocks, capturing every tool call (in order) and the final text.
- **Tool order check** — for tasks where the order matters (e.g. `list_files` must come before `read_file`), the evaluator walks the actual call sequence and checks that expected tools appear in the right order.
- **LLM judge** — a separate model call grades the final response on a 1-10 scale, given the original task, the tool results, and the agent's text.

## Code

### `evals/mocks/tools.py` (NEW) — mock tool factories

```python
from typing import Any


def create_mock_read_file(mock_content: str):
    def execute(args: dict[str, Any]) -> str:
        return mock_content
    return execute


def create_mock_write_file(mock_response: str = None):
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        content = args.get("content", "")
        path = args.get("path", "unknown")
        return f"Successfully wrote {len(content)} characters to {path}"
    return execute


def create_mock_list_files(mock_files: list[str]):
    def execute(args: dict[str, Any]) -> str:
        return "\n".join(mock_files)
    return execute


def create_mock_delete_file(mock_response: str = None):
    def execute(args: dict[str, Any]) -> str:
        if mock_response:
            return mock_response
        return f"Successfully deleted {args.get('path', 'unknown')}"
    return execute


def create_mock_shell(mock_output: str):
    def execute(args: dict[str, Any]) -> str:
        return mock_output
    return execute
```

### `evals/types.py` — add multi-turn types

```python
@dataclass
class MockToolConfig:
    description: str
    parameters: dict[str, str]
    mock_return: str


@dataclass
class MultiTurnEvalData:
    mock_tools: dict[str, MockToolConfig]
    prompt: Optional[str] = None
    messages: Optional[list[dict[str, Any]]] = None
    config: Optional[dict[str, Any]] = None


@dataclass
class MultiTurnTarget:
    original_task: str
    mock_tool_results: dict[str, str]
    category: str
    expected_tool_order: Optional[list[str]] = None
    forbidden_tools: Optional[list[str]] = None


@dataclass
class MultiTurnResult:
    text: str
    steps: list[dict[str, Any]]
    tools_used: list[str]
    tool_call_order: list[str]
```

### `evals/utils.py` — add `build_mocked_tools`

```python
def build_mocked_tools(
    mock_tools: dict[str, dict[str, Any]],
) -> tuple[list[dict], dict[str, callable]]:
    """Build OpenAI tool definitions and executors from mock config."""
    tool_definitions = []
    executor_map = {}

    for name, config in mock_tools.items():
        properties = {}
        for param_name in config["parameters"]:
            properties[param_name] = {"type": "string"}

        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": config["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        tool_definitions.append(tool_def)

        mock_return = config["mock_return"]
        executor_map[name] = lambda args, ret=mock_return: ret

    return tool_definitions, executor_map
```

### `evals/executors.py` — add `multi_turn_with_mocks`

```python
def multi_turn_with_mocks(data: dict[str, Any]) -> MultiTurnResult:
    tool_definitions, executor_map = build_mocked_tools(data["mock_tools"])

    if "messages" in data and data["messages"]:
        messages = data["messages"]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data["prompt"]},
        ]

    model = "gpt-5-mini"
    max_steps = 20
    if data.get("config"):
        model = data["config"].get("model", model)
        max_steps = data["config"].get("max_steps", max_steps)

    all_tool_calls: list[str] = []
    steps: list[dict[str, Any]] = []
    final_text = ""

    for step_num in range(max_steps):
        response = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        step_data: dict[str, Any] = {}

        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)
                all_tool_calls.append(tool_name)

                executor = executor_map.get(tool_name)
                result = executor(args) if executor else f"Unknown tool: {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        if message.content:
            final_text = message.content

        steps.append(step_data)

        if finish_reason != "tool_calls":
            messages.append({"role": "assistant", "content": message.content or ""})
            break

    return MultiTurnResult(
        text=final_text,
        steps=steps,
        tools_used=list(set(all_tool_calls)),
        tool_call_order=all_tool_calls,
    )
```

### `evals/evaluators.py` — add `tool_order_correct` and `llm_judge`

See the [done branch](https://github.com/sivakarasala/building-ai-agents-python/blob/done/evals/evaluators.py) for the full code.

### `evals/agent_multiturn_eval.py` and `evals/data/agent_multiturn.json` — see [done branch](https://github.com/sivakarasala/building-ai-agents-python/tree/done/evals)

## Exercises

1. Write a multi-turn case where the agent must `list_files`, then `read_file`, then summarize. Score it with the LLM judge.
2. Write a *negative* multi-turn case where the agent should NOT call any tool — verify with `tools_avoided`.
3. Compare the LLM judge's score to your own gut score on 5 cases. Where does it disagree?
