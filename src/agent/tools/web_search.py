from typing import Any

# Web search is a provider-managed tool — OpenAI handles execution.
# We just define it so the API knows to enable it.
WEB_SEARCH_TOOL = {
    "type": "web_search_preview",
}


def web_search_execute(args: dict[str, Any]) -> str:
    """Provider tools are executed by OpenAI, not us."""
    return "Provider tool web_search - executed by model provider"
