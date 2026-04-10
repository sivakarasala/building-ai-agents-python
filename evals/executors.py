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
