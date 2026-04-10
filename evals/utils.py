from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data.

    Returns a Responses API input list. The system prompt is also returned in
    the array (as a system message) so existing tests that index msgs[0] /
    msgs[1] keep working — single_turn_executor pulls it out and passes it via
    `instructions` instead.
    """
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]
