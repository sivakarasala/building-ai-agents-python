from src.agent.execute_tool import execute_tool
from src.agent.tools import ALL_TOOLS, TOOL_EXECUTORS


def test_execute_tool_unknown():
    out = execute_tool("nope", {})
    assert "Unknown tool" in out


def test_registry_starts_empty():
    # File tools will be added during this lesson.
    assert TOOL_EXECUTORS == {}
    assert ALL_TOOLS == []
