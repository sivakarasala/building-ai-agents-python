# Lesson 9 — Human-in-the-Loop Tool Approval

## Overview

By default the agent executes every tool call the model decides on. That's fine for `read_file` but very much not fine for `delete_file`, `run_command`, or `execute_code`. In this lesson you wire a synchronous approval gate into the agent loop so a human can approve or reject each tool call before it runs.

## Key concepts

- The agent loop is sync, but the approval prompt is naturally async (you wait for a UI event). The trick is to support both shapes via `inspect.isawaitable`.
- The UI helper (`src/ui/tool_approval.py`) and the `on_tool_approval` callback field on `AgentCallbacks` already exist — they're just not called yet.
- Rejecting a tool breaks out of the loop entirely. The agent does not silently skip and continue.

## Code

### `src/agent/run.py` — wire approval into the tool execution loop

Replace the existing tool execution block:

```python
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
```

with:

```python
        # Execute each tool and add results to message history
        import asyncio
        import inspect

        rejected = False
        for tc in tool_calls:
            # Check for approval (callback may be sync or async)
            approval = callbacks.on_tool_approval(tc.tool_name, tc.args)
            if inspect.isawaitable(approval):
                approved = asyncio.run(approval)
            else:
                approved = approval

            if not approved:
                rejected = True
                break

            result = execute_tool(tc.tool_name, tc.args)
            callbacks.on_tool_call_end(tc.tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "content": result,
            })

            report_token_usage()

        if rejected:
            break
```

### What's already in place (do NOT need to add)

- `src/types.py` already defines `ToolApprovalRequest` and the `on_tool_approval` field on `AgentCallbacks`.
- `src/ui/tool_approval.py` already provides the `request_approval(name, args)` helper that prints the proposed call and reads y/n from stdin.
- `src/ui/app.py` already wires its `on_tool_approval` callback to `request_approval`.

## Exercises

1. **Auto-approve safe tools.** Wire approval only for `delete_file`, `run_command`, `execute_code`. The rest pass through.
2. **Remember the answer.** If the user says "approve all `read_file`", subsequent `read_file` calls in the same session skip the prompt.
3. **Reject reason.** When a user rejects, write the reason back into the conversation as a tool message so the model can react.
