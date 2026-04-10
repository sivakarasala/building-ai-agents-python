# Lesson 7 — Web Search + Context Management

## Overview

Two new things in one lesson, because they're both about *the agent's relationship to context*.

1. **Web search** — give the agent a way to look up things outside its training data via OpenAI's provider-managed `web_search_preview` tool. (No execution; the model handles it.)
2. **Context management** — when the conversation grows past the model's context window threshold, summarize old messages so the agent can keep going.

## Key concepts

- The web search tool has a different shape from function tools — it's `{"type": "web_search_preview"}`. The Chat Completions API doesn't currently accept it directly, so the agent loop filters tools by `type == "function"` before passing them to `chat.completions.create`. Add it to the registry but understand it would need the Responses API to actually run.
- The threshold check uses `is_over_threshold(total, context_window)` which compares to `DEFAULT_THRESHOLD * context_window` (default 0.8 = 80%).
- Compaction creates a fresh user/assistant pair: a `[CONVERSATION SUMMARY]` user message and an "I understand, let's continue" acknowledgment from the assistant. The original messages are dropped.
- Token usage callbacks fire on every message change so the UI can show a progress bar.

## Code

### `src/agent/tools/web_search.py` (NEW)

```python
from typing import Any

WEB_SEARCH_TOOL = {
    "type": "web_search_preview",
}


def web_search_execute(args: dict[str, Any]) -> str:
    """Provider tools are executed by OpenAI, not us."""
    return "Provider tool web_search - executed by model provider"
```

### `src/agent/tools/__init__.py` — wire web_search into the registry

```python
from src.agent.tools.web_search import WEB_SEARCH_TOOL, web_search_execute
```

Add `"web_search": web_search_execute` to `TOOL_EXECUTORS` and `WEB_SEARCH_TOOL` to `ALL_TOOLS`.

### `src/agent/context/model_limits.py` — replace the stubs

```python
def is_over_threshold(
    total_tokens: int,
    context_window: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Check if token usage exceeds the threshold."""
    return total_tokens > context_window * threshold


def calculate_usage_percentage(total_tokens: int, context_window: int) -> float:
    """Calculate usage percentage."""
    return (total_tokens / context_window) * 100
```

### `src/agent/context/compaction.py` — replace the stub

```python
from typing import Any
from openai import OpenAI
from src.agent.context.token_estimator import extract_message_text

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary of the conversation so far that preserves:

1. Key decisions and conclusions reached
2. Important context and facts mentioned
3. Any pending tasks or questions
4. The overall goal of the conversation

Be concise but complete. The summary should allow the conversation to continue naturally.

Conversation to summarize:
"""


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Format messages as readable text for summarization."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = extract_message_text(msg)
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def compact_conversation(
    messages: list[dict[str, Any]],
    model: str = "gpt-5-mini",
) -> list[dict[str, Any]]:
    """Compact a conversation by summarizing it with an LLM."""
    conversation_messages = [m for m in messages if m.get("role") != "system"]

    if not conversation_messages:
        return []

    conversation_text = messages_to_text(conversation_messages)

    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": SUMMARIZATION_PROMPT + conversation_text}
        ],
    )

    summary = response.choices[0].message.content

    return [
        {
            "role": "user",
            "content": (
                f"[CONVERSATION SUMMARY]\n"
                f"The following is a summary of our conversation so far:\n\n"
                f"{summary}\n\n"
                f"Please continue from where we left off."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "I understand. I've reviewed the summary of our conversation "
                "and I'm ready to continue. How can I help you next?"
            ),
        },
    ]
```

### `src/agent/run.py` — wire compaction + token reporting into the loop

Add to the imports:

```python
from src.agent.context import (
    estimate_messages_tokens,
    get_model_limits,
    is_over_threshold,
    calculate_usage_percentage,
    compact_conversation,
    DEFAULT_THRESHOLD,
)
from src.types import AgentCallbacks, ToolCallInfo, TokenUsageInfo
```

At the top of `run_agent`, before building `messages`:

```python
    model_limits = get_model_limits(MODEL_NAME)

    working_history = filter_compatible_messages(conversation_history)
    pre_check_tokens = estimate_messages_tokens([
        {"role": "system", "content": SYSTEM_PROMPT},
        *working_history,
        {"role": "user", "content": user_message},
    ])

    if is_over_threshold(pre_check_tokens.total, model_limits.context_window):
        working_history = compact_conversation(working_history, MODEL_NAME)
```

After `messages` is built, define and call `report_token_usage`:

```python
    def report_token_usage():
        if callbacks.on_token_usage:
            usage = estimate_messages_tokens(messages)
            callbacks.on_token_usage(TokenUsageInfo(
                input_tokens=usage.input,
                output_tokens=usage.output,
                total_tokens=usage.total,
                context_window=model_limits.context_window,
                threshold=DEFAULT_THRESHOLD,
                percentage=calculate_usage_percentage(
                    usage.total, model_limits.context_window
                ),
            ))

    report_token_usage()
```

And add `report_token_usage()` calls after each tool result is appended inside the loop.

## Exercises

1. Lower `DEFAULT_THRESHOLD` to 0.3 and watch compaction kick in earlier. Look at the summary the model produces.
2. Try a long multi-turn conversation about an unrelated topic, then ask the agent something that depends on the early messages — does the summary preserve enough?
3. Wire `web_search` to actually run. Hint: switch the agent loop to the OpenAI Responses API for that single call.
