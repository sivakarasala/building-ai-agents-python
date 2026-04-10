from typing import Any
from src.agent.system.prompt import SYSTEM_PROMPT


def build_messages(
    data: dict[str, Any],
) -> list[dict[str, str]]:
    """Build message array from eval data."""
    system_prompt = data.get("system_prompt") or SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["prompt"]},
    ]
