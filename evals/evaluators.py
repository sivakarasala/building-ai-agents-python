from typing import Union
from evals.types import EvalTarget, SingleTurnResult


def tools_selected(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Check if all expected tools were selected. Returns 1 or 0."""
    expected = target.expected_tools
    if not expected:
        return 1.0

    selected = set(output.tool_names)
    return 1.0 if all(t in selected for t in expected) else 0.0


def tools_avoided(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Check if forbidden tools were avoided. Returns 1 or 0."""
    forbidden = target.forbidden_tools
    if not forbidden:
        return 1.0

    selected = set(output.tool_names)
    return 0.0 if any(t in selected for t in forbidden) else 1.0


def tool_selection_score(
    output: SingleTurnResult,
    target: EvalTarget,
) -> float:
    """Precision/recall F1 score for tool selection. Returns 0 to 1."""
    if not target.expected_tools:
        return 0.5 if output.selected_any else 1.0

    expected = set(target.expected_tools)
    selected = set(output.tool_names)

    hits = len([t for t in output.tool_names if t in expected])
    precision = hits / len(selected) if selected else 0.0
    recall = hits / len(expected) if expected else 0.0

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
