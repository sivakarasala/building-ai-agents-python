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

### `src/agent/system/prompt.py`

```python
SYSTEM_PROMPT = """You are a helpful AI assistant. You provide clear, accurate, and concise responses to user questions.

Guidelines:
- Be direct and helpful
- If you don't know something, say so honestly
- Provide explanations when they add value
- Stay focused on the user's actual question"""
```

### `src/agent/run.py`

The simplest possible "agent" — one LLM call, no tools, no loop:

```python
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.system.prompt import SYSTEM_PROMPT
from src.types import AgentCallbacks

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
    """Single-shot LLM call. Not a real agent yet — no tools, no loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": user_message},
    ]

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    text = response.choices[0].message.content or ""
    callbacks.on_token(text)
    callbacks.on_complete(text)

    return [*messages, {"role": "assistant", "content": text}]
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
4. **Print the raw response object.** Look at `usage`, `finish_reason`, and the message structure — these are the building blocks for the rest of the course.
