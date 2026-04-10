# Lesson 4 — The Agent Loop

## Overview

This is the heart of the course. You build the agent loop: a `while True:` that streams a model response, intercepts tool calls, executes them, feeds the results back, and loops until the model produces a final text answer with no more tool calls.

We use OpenAI's **Responses API** (`client.responses.create`) — the newer, recommended path. It's simpler than Chat Completions for tool-using agents because tool calls and tool outputs are first-class typed items in the conversation history, not parallel arrays you have to keep in sync.

## Key concepts

- **Responses API vs Chat Completions.** Responses API takes `input=[items...]` instead of `messages=[...]`. The system prompt is passed via the `instructions` parameter, not as a system message in the input. Tool definitions are *flat*: `{"type": "function", "name": ..., "parameters": ...}` — no nested `"function": {...}` wrapper.
- **Input items.** History is a list of typed items: `{"role": "user"|"assistant", "content": "..."}` for plain messages, `{"type": "function_call", "call_id": ..., "name": ..., "arguments": ...}` when the model calls a tool, and `{"type": "function_call_output", "call_id": ..., "output": ...}` when you return the result.
- **Streaming.** The stream yields events. `response.output_text.delta` events carry text chunks. The final `response.completed` event hands you the full `response` object — its `output` array contains every item the model produced this turn (text, function_call, reasoning, etc.). Much simpler than reassembling fragmented `delta.tool_calls` from Chat Completions.
- **Looping.** Walk `response.output`, append every item to `input_items` (so the next turn has full context), collect any `function_call` items, execute them, and append a matching `function_call_output` for each. If there were no function calls, you're done.
- **Callbacks.** The loop is sync but it talks to the UI through the `AgentCallbacks` dataclass: `on_token` (stream tokens), `on_tool_call_start`, `on_tool_call_end`, `on_complete`.

## Code

Create `src/agent/run.py`:

```python
import json
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import ALL_TOOLS
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
    """Run the agent loop using the OpenAI Responses API.

    Conversation history is a list of Responses API "input items":
      - {"role": "user"|"assistant", "content": "..."}
      - {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
      - {"type": "function_call_output", "call_id": "...", "output": "..."}

    The system prompt is sent via the `instructions` parameter, not as a message.
    """
    working_history = filter_compatible_messages(conversation_history)

    input_items: list[dict[str, Any]] = [
        *working_history,
        {"role": "user", "content": user_message},
    ]

    full_response = ""

    while True:
        stream = _get_client().responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=ALL_TOOLS if ALL_TOOLS else None,
            stream=True,
        )

        # Stream text deltas to the UI; capture the final response object on
        # `response.completed` so we can read its full output items.
        final_response = None
        current_text = ""

        for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    current_text += delta
                    callbacks.on_token(delta)

            elif event_type == "response.completed":
                final_response = getattr(event, "response", None)

        full_response += current_text

        if final_response is None:
            # Stream ended without a completed event — nothing more to do
            break

        # Walk the output items: append everything (assistant text, reasoning,
        # function_call) to history so the next turn has full context, and
        # collect any function_call items we need to execute.
        function_calls: list[ToolCallInfo] = []

        for item in final_response.output:
            item_dict = item.model_dump(exclude_none=True)
            input_items.append(item_dict)

            if item_dict.get("type") == "function_call":
                try:
                    args = json.loads(item_dict.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                function_calls.append(ToolCallInfo(
                    tool_call_id=item_dict["call_id"],
                    tool_name=item_dict["name"],
                    args=args,
                ))

        # No function calls → the model gave a final answer; we're done
        if not function_calls:
            break

        for tc in function_calls:
            callbacks.on_tool_call_start(tc.tool_name, tc.args)

        # Execute each function call and append the corresponding
        # function_call_output item back into the input.
        for tc in function_calls:
            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            input_items.append({
                "type": "function_call_output",
                "call_id": tc.tool_call_id,
                "output": result,
            })

    callbacks.on_complete(full_response)
    return input_items
```

## Exercises

1. **Cap iterations.** Add a `max_iterations=10` guard so a misbehaving model can't infinite-loop.
2. **Print every iteration.** Add a debug print at the top of the `while True` showing iteration number + which tools were just executed.
3. **Refuse repeats.** Detect when the model calls the same tool with the same args twice in a row and break early.
