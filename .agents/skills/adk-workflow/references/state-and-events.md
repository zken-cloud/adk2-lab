# State and Events Reference

Manage shared state across workflow nodes and understand the event system.

## Workflow Context

Every node receives a `Context` object (when declaring a `ctx` parameter):

```python
from google.adk.agents.context import Context

def my_node(ctx: Context, node_input: str) -> str:
  # Access shared state
  value = ctx.state.get("key", "default")

  # Write to state
  ctx.state["key"] = "new_value"

  # Access session info
  session_id = ctx.session.id
  invocation_id = ctx.invocation_id

  # Get node metadata
  node_path = ctx.node_path        # e.g., "MyWorkflow/my_node"
  triggered_by = ctx.triggered_by  # Name of predecessor node
  retry_count = ctx.retry_count    # 0 on first attempt

  return f"Processed: {value}"
```

## Context Properties

| Property | Type | Description |
|----------|------|-------------|
| `node_path` | `str` | Full path of current node (e.g., "WorkflowA/node1") |
| `execution_id` | `str` | Unique ID for this execution |
| `triggered_by` | `str` | Name of node that triggered current node |
| `in_nodes` | `frozenset[str]` | Names of all predecessor nodes |
| `resume_inputs` | `dict[str, Any]` | Inputs for resuming (keyed by interrupt_id) |
| `retry_count` | `int` | Number of times this node has been retried |
| `session` | `Session` | Current session (with local events merged) |
| `state` | `dict` | Shared workflow state |
| `invocation_id` | `str` | Current invocation ID |

## State Management

State is shared across all nodes in a workflow invocation. **Prefer `Event(state=...)` over `ctx.state[...] =`** for setting state:

```python
# ✅ Preferred: set state via Event (persisted in event history, replayable)
def node_a(node_input: str):
  return Event(
      data="done",
      state={"user_data": {"name": "Alice", "score": 95}},
  )

# ❌ Avoid: direct ctx.state mutation (not captured in event history)
def node_a(ctx: Context, node_input: str) -> str:
  ctx.state["user_data"] = {"name": "Alice", "score": 95}
  return "done"
```

**Why `Event(state=...)` is preferred:**
- State deltas are persisted in event history as `event.actions.state_delta`
- Non-resumable HITL can reconstruct state by replaying events
- Makes state changes explicit and traceable
- `ctx.state` mutations are side effects that may be lost on replay

Reading state is always done via `ctx.state`:

```python
def node_b(ctx: Context, node_input: str) -> str:
  user = ctx.state["user_data"]
  return f"User {user['name']} scored {user['score']}"
```

The `state` dict is stored as `event.actions.state_delta` and applied to the session.

## State as Function Parameters

FunctionNode automatically resolves parameters from state:

```python
# If ctx.state["user_name"] = "Alice" and ctx.state["threshold"] = 0.5
def my_node(node_input: str, user_name: str, threshold: float) -> str:
  # user_name = "Alice" (from state)
  # threshold = 0.5 (from state)
  return f"{user_name}: {node_input} (threshold={threshold})"
```

Resolution order:
1. `ctx` -> Context object
2. `node_input` -> predecessor output
3. Other names -> `ctx.state[param_name]` (with auto type conversion)
4. Default values if not in state

## Event Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Any` | Output data passed to downstream nodes |
| `route` | `str\|bool\|int\|list` | Routing signal for conditional edges |
| `state` | `dict` (constructor only) | State delta to apply |
| `content` | `types.Content` | Content for intermediate display |
| `no_trigger` | `bool` | If True, don't trigger downstream nodes |
| `node_path` | `str` | Set by workflow (e.g., "Workflow/node_name") |
| `execution_id` | `str` | Set by workflow for tracking |
| `is_terminal_output` | `bool` | True if node has no downstream edges |

## Workflow Data Rules

- **`Event.data` must be JSON-serializable.** FunctionNode auto-converts Pydantic `BaseModel` returns via `model_dump()`, so returning a model is safe. But `types.Content` and other non-serializable objects will fail with SQLite/database session services.
- **`output_key` stores dicts, not BaseModel instances.** LLM agents with `output_schema` use `validate_schema()` → `model_dump()` internally, so `ctx.state[output_key]` is always a plain dict.
- **`ctx.state.get(key)` returns a dict.** Use dict access (`data["field"]`) or reconstruct the model (`MyModel(**data)`) if you need typed access.

```python
# Reading output_key from state — it's a dict, not a BaseModel
def use_plan(ctx: Context, node_input: Any) -> str:
  plan = ctx.state.get('task_plan', {})  # dict, not TaskPlan
  return plan['project_name']            # dict access

  # Or reconstruct if you need typed access:
  plan_model = TaskPlan(**plan)
  return plan_model.project_name
```

## Content Events (User-Visible Output)

In the ADK web UI, only `event.content` is rendered — `event.data` is internal and not displayed. Emit content events for any user-facing output:

```python
from google.genai import types
from google.adk.events.event import Event as AdkEvent

async def verbose_node(ctx: Context, node_input: str):
  # Emit intermediate progress
  yield AdkEvent(
      author="verbose_node",
      invocation_id=ctx.invocation_id,
      content=types.Content(
          role="model",
          parts=[types.Part.from_text("Processing step 1...")],
      ),
  )

  # Emit actual output
  yield Event(data="final result")
```

## No-Trigger Events

Emit an event that doesn't trigger downstream nodes (used internally by JoinNode while waiting):

```python
yield Event(no_trigger=True)  # Downstream nodes won't fire
```

## Terminal Outputs

Events from nodes with no outgoing edges have `is_terminal_output=True` automatically. These are the final outputs of the workflow.
