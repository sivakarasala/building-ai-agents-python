import json
from typing import Any
from openai import OpenAI
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from evals.types import EvalData, SingleTurnResult, MultiTurnEvalData, MultiTurnResult
from evals.utils import build_messages, build_mocked_tools

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

    # Filter to only tools specified in data
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

    # Extract tool calls
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


def multi_turn_with_mocks(data: dict[str, Any]) -> MultiTurnResult:
    """Run a multi-turn evaluation with mocked tools."""
    tool_definitions, executor_map = build_mocked_tools(data["mock_tools"])

    # Build messages
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

        # Process tool calls
        if message.tool_calls:
            step_tool_calls = []
            step_tool_results = []

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)
                all_tool_calls.append(tool_name)

                step_tool_calls.append({
                    "tool_name": tool_name,
                    "args": args,
                })

                # Execute mock tool
                executor = executor_map.get(tool_name)
                result = executor(args) if executor else f"Unknown tool: {tool_name}"

                step_tool_results.append({
                    "tool_name": tool_name,
                    "result": result,
                })

                # Add tool result to history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            step_data["tool_calls"] = step_tool_calls
            step_data["tool_results"] = step_tool_results

        # Process text
        if message.content:
            step_data["text"] = message.content
            final_text = message.content

        steps.append(step_data)

        # Stop if no tool calls (LLM is done)
        if finish_reason != "tool_calls":
            messages.append({
                "role": "assistant",
                "content": message.content or "",
            })
            break

    tools_used = list(set(all_tool_calls))

    return MultiTurnResult(
        text=final_text,
        steps=steps,
        tools_used=tools_used,
        tool_call_order=all_tool_calls,
    )
