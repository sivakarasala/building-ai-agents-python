# Lesson 4 — The Agent Loop

## Overview

This is the heart of the course. You build the agent loop: a `while True:` that streams a model response, intercepts tool calls, executes them, feeds the results back, and loops until the model produces a final text answer with no more tool calls.

## Key concepts

- **Streaming.** The OpenAI client returns a stream of chunks. Text content comes in `delta.content`; tool calls come in fragments via `delta.tool_calls`. You accumulate both.
- **Tool call assembly.** The model can request multiple tool calls in one assistant turn. Each call's `id`, `name`, and `arguments` arrive in fragments across many chunks — you assemble them by index.
- **Two message types per round.** When tools are called: (1) one assistant message with `content` + `tool_calls` array, (2) one `tool` message *per tool call* with the result keyed by `tool_call_id`.
- **Finish reason.** If `finish_reason != "tool_calls"` (or there are no tool calls), the model is done — append the final text and break.
- **Callbacks.** The loop is sync but it talks to the UI through the `AgentCallbacks` dataclass: `on_token` (stream tokens), `on_tool_call_start`, `on_tool_call_end`, `on_complete`.

## Code

Replace the stub in `src/agent/run.py` with this:

```python
import json
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from src.agent.execute_tool import execute_tool
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.system.filter_messages import filter_compatible_messages
from src.types import AgentCallbacks, ToolCallInfo

load_dotenv()

_client: OpenAI | None = None
MODEL_NAME = "gpt-5-mini"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def run_agent(
    user_message: str,
    conversation_history: list[dict[str, Any]],
    callbacks: AgentCallbacks,
) -> list[dict[str, Any]]:
    """Run the agent loop. Returns the updated message history."""

    working_history = filter_compatible_messages(conversation_history)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *working_history,
        {"role": "user", "content": user_message},
    ]

    full_response = ""

    while True:
        chat_tools = [t for t in ALL_TOOLS if t.get("type") == "function"]
        stream = _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=chat_tools if chat_tools else None,
            stream=True,
        )

        current_text = ""
        tool_calls_data: dict[int, dict[str, Any]] = {}
        finish_reason = None

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                current_text += delta.content
                callbacks.on_token(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_data[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[idx]["arguments"] += tc_delta.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        full_response += current_text

        # Build tool calls list
        tool_calls: list[ToolCallInfo] = []
        for idx in sorted(tool_calls_data.keys()):
            tc_data = tool_calls_data[idx]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(ToolCallInfo(
                tool_call_id=tc_data["id"],
                tool_name=tc_data["name"],
                args=args,
            ))
            callbacks.on_tool_call_start(tc_data["name"], args)

        # If no tool calls, we're done
        if finish_reason != "tool_calls" or not tool_calls:
            messages.append({"role": "assistant", "content": current_text or ""})
            break

        # Add the assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": current_text or None,
            "tool_calls": [
                {
                    "id": tc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute each tool and add results to message history
        for tc in tool_calls:
            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "content": result,
            })

    callbacks.on_complete(full_response)
    return messages
```

## Exercises

1. **Cap iterations.** Add a `max_iterations=10` guard so a misbehaving model can't infinite-loop.
2. **Print every iteration.** Add a debug print at the top of the `while True` showing iteration number + which tools were just executed.
3. **Refuse repeats.** Detect when the model calls the same tool with the same args twice in a row and break early.
