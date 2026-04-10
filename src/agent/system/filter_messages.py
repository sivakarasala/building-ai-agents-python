from typing import Any


def filter_compatible_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter conversation history to only include compatible message formats.

    Provider tools (like web_search) may return messages with formats that
    cause issues when passed back to subsequent API calls.
    """
    filtered = []
    for msg in messages:
        role = msg.get("role")

        # Always keep user and system messages
        if role in ("user", "system"):
            filtered.append(msg)
            continue

        # Keep tool messages
        if role == "tool":
            filtered.append(msg)
            continue

        # Keep assistant messages that have text content or tool calls
        if role == "assistant":
            content = msg.get("content")
            has_text = isinstance(content, str) and content.strip()
            has_tool_calls = bool(msg.get("tool_calls"))

            if has_text or has_tool_calls:
                filtered.append(msg)
                continue

    return filtered
