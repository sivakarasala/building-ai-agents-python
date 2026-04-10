# Lesson 1 — Intro to Agents

## Overview

This lesson introduces AI agents — what they are, why they matter, and where they excel (or fail). Then you write the simplest possible "agent": a one-shot LLM call that takes a user message and prints a response. It is barely an agent, but it's the seed everything in the rest of the course grows from.

## What is an agent?

Everyone has a different definition. Mine:

> **An agent is an LLM that can take actions in a loop until a task is complete.**

Three pieces:

1. **LLM** — a language model that can reason and decide.
2. **Actions** — tools it can call to affect the world (read files, run shell, hit APIs).
3. **Loop** — keeps going until the task is done, not just one response.

A chatbot responds once. An agent keeps working. The common thread across every definition: **the LLM is in the driver's seat**, picking what to do and when to stop.

## Why agents?

LLMs alone are limited to:
- Stale knowledge from training
- Single-turn responses
- Generating text — no real-world impact

Agents can:
- Pull live data (APIs, DBs, web)
- Work through multi-step problems
- Actually *do* things (write files, send emails, deploy code)
- Recover from errors and try again

The difference is **agency** — the ability to act on the world, not just describe it.

## What agents are good at

1. Repetitive knowledge work (research, summarization)
2. Code: writing, debugging, refactoring
3. Multi-step workflows that chain several tools
4. Exploration: "find every X in this codebase and do Y"
5. Assisting humans, not replacing them

## What agents are bad at

1. Anything requiring physical presence
2. High-stakes irreversible decisions (don't let an agent fire someone)
3. Creative work that needs human taste — they can draft, you decide
4. Tasks with ambiguous success criteria
5. Real-time / latency-sensitive work — LLM calls are slow
6. True logical reasoning — they pattern-match and confidently get edge cases wrong

The biggest failure mode: **an agent confidently doing the wrong thing**. They don't know what they don't know.

## The agent loop (preview)

Every agent in this course follows the same pattern:

```
1. Receive task
2. Think about what to do
3. Take an action (call a tool) or respond
4. Observe the result
5. If not done, go to step 2
```

This is sometimes called **ReAct** (Reason + Act). We will build this loop in lesson 4. For now, we just need the simplest possible LLM call.

## Code

### `src/main.py`

The simplest possible "agent" — one LLM call, no tools, no loop. We use OpenAI's **Responses API** (`client.responses.create`) from the start — it's the modern, recommended path and the same one we'll build the agent loop on in lesson 4.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {"role": "user", "content": "What is an AI agent in one sentence?"}
    ],
)

print(response.output_text)
```

Run it:

```bash
python -m src.main
```

A couple of Responses-API-specific things to notice:

- The conversation lives in **`input`** (a list of "input items"), not `messages`.
- The convenience **`output_text`** field concatenates all assistant text from the response — much simpler than `response.choices[0].message.content`.
- The system prompt (which we'll add in a moment) is passed via the **`instructions`** parameter, not as a `{"role": "system", ...}` item.

### `src/agent/system/prompt.py`

Agents need personality and guidelines. We won't wire it in yet — that comes in lesson 4 when we build the actual agent loop — but defining it now keeps the file structure stable:

```python
SYSTEM_PROMPT = """You are a helpful AI assistant. You provide clear, accurate, and concise responses to user questions.

Guidelines:
- Be direct and helpful
- If you don't know something, say so honestly
- Provide explanations when they add value
- Stay focused on the user's actual question"""
```

### `src/types.py`

The core data structures that the agent and UI will share. We won't use all of them immediately, but defining them now gives a clear picture of where we're headed:

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional


@dataclass
class ToolCallInfo:
    """Metadata about a tool the LLM wants to call."""
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class ModelLimits:
    """Token limits for a model."""
    input_limit: int
    output_limit: int
    context_window: int


@dataclass
class TokenUsageInfo:
    """Current token usage for display."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_window: int
    threshold: float
    percentage: float


@dataclass
class AgentCallbacks:
    """How the agent communicates back to the UI."""
    on_token: Callable[[str], None]
    on_tool_call_start: Callable[[str, Any], None]
    on_tool_call_end: Callable[[str, str], None]
    on_complete: Callable[[str], None]
    on_tool_approval: Callable[[str, Any], Awaitable[bool]]
    on_token_usage: Optional[Callable[[TokenUsageInfo], None]] = None


@dataclass
class ToolApprovalRequest:
    """A pending tool approval for the UI to display."""
    tool_name: str
    args: Any
    resolve: Callable[[bool], None]
```

This is barely an agent because:
- No tools — it can't take actions
- No loop — it responds once and stops
- No tool execution — there's nothing to observe

We will fix all three in lessons 2-4.

## Key points

1. **Agents = LLM + Actions + Loop.** The model decides what to do and keeps going.
2. **The LLM controls the flow.** It is not just answering, it is driving.
3. **Agents are best at repetitive knowledge work.** They are bad at ambiguous, high-stakes, or true-reasoning tasks.
4. **Start dumb.** A single LLM call is the foundation. We will add capabilities incrementally.

## Exercises

1. **Run it.** Send a message and see the response.
2. **Change the model.** Try `gpt-4o-mini` vs `gpt-5-mini` — compare quality and latency.
3. **Tweak the system prompt.** Make it terse, make it verbose, make it pretend to be a pirate. Notice how much behavior comes from the prompt.
4. **Print the raw response object.** Look at `response.usage`, `response.output` (the typed output items), and `response.output_text` — these are the building blocks for the rest of the course.
