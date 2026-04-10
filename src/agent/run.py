from typing import Any
from src.types import AgentCallbacks

MODEL_NAME = "gpt-5-mini"


def run_agent(
    user_message: str,
    conversation_history: list[dict[str, Any]],
    callbacks: AgentCallbacks,
) -> list[dict[str, Any]]:
    """Run the agent loop. Returns the updated message history.

    STUB — you'll build this during this lesson.
    """
    callbacks.on_complete("(agent loop not implemented yet)")
    return conversation_history
