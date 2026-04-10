import json
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS
from src.agent.execute_tool import execute_tool
from src.agent.system.prompt import SYSTEM_PROMPT
from src.agent.context import (
    estimate_messages_tokens,
    get_model_limits,
    is_over_threshold,
    calculate_usage_percentage,
    compact_conversation,
    DEFAULT_THRESHOLD,
)
from src.agent.system.filter_messages import filter_compatible_messages
from src.types import AgentCallbacks, ToolCallInfo, TokenUsageInfo

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

    model_limits = get_model_limits(MODEL_NAME)

    # Filter and check if we need to compact
    working_history = filter_compatible_messages(conversation_history)
    pre_check_tokens = estimate_messages_tokens([
        {"role": "system", "content": SYSTEM_PROMPT},
        *working_history,
        {"role": "user", "content": user_message},
    ])

    if is_over_threshold(pre_check_tokens.total, model_limits.context_window):
        working_history = compact_conversation(working_history, MODEL_NAME)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *working_history,
        {"role": "user", "content": user_message},
    ]

    # Report token usage
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

    full_response = ""

    while True:
        # Chat Completions API only accepts function/custom tools.
        # Provider-managed tools (e.g. web_search_preview) are filtered out here;
        # they would need the Responses API to be active.
        chat_tools = [t for t in ALL_TOOLS if t.get("type") == "function"]
        stream = _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=chat_tools if chat_tools else None,
            stream=True,
        )

        # Accumulate the streamed response
        current_text = ""
        tool_calls_data: dict[int, dict[str, Any]] = {}
        current_role = "assistant"

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # Accumulate text content
            if delta.content:
                current_text += delta.content
                callbacks.on_token(delta.content)

            # Accumulate tool calls (they arrive in fragments)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        tool_calls_data[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_data[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_data[idx]["arguments"] += (
                                tc_delta.function.arguments
                            )

            # Check for finish reason
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

            tool_calls.append(
                ToolCallInfo(
                    tool_call_id=tc_data["id"],
                    tool_name=tc_data["name"],
                    args=args,
                )
            )
            callbacks.on_tool_call_start(tc_data["name"], args)

        # If no tool calls, we're done
        if finish_reason != "tool_calls" or not tool_calls:
            # Add the assistant's text response to history
            messages.append({"role": "assistant", "content": current_text or ""})
            break

        # Add the assistant message with tool calls to history
        assistant_message: dict[str, Any] = {
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
        }
        messages.append(assistant_message)

        # Execute each tool and add results to message history
        for tc in tool_calls:
            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "content": result,
            })

            report_token_usage()

    callbacks.on_complete(full_response)
    return messages
