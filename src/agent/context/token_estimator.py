import json
from typing import Any
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate token count using character division.
    Uses 3.75 as the divisor (midpoint of 3.5-4 range).
    """
    return max(1, len(text) // 4 + 1)


def extract_message_text(message: dict[str, Any]) -> str:
    """Extract text content from a message."""
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "value" in part:
                    parts.append(str(part["value"]))
                else:
                    parts.append(json.dumps(part))
        return " ".join(parts)

    if content is None:
        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            return json.dumps(tool_calls)
        return ""

    return json.dumps(content)


@dataclass
class TokenUsage:
    input: int
    output: int
    total: int


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> TokenUsage:
    """Estimate token counts for a message array.
    Separates input (user, system, tool) from output (assistant).
    """
    input_tokens = 0
    output_tokens = 0

    for message in messages:
        text = extract_message_text(message)
        tokens = estimate_tokens(text)

        if message.get("role") == "assistant":
            output_tokens += tokens
        else:
            input_tokens += tokens

    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        total=input_tokens + output_tokens,
    )
