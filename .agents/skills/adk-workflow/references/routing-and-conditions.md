# Routing and Conditional Branching Reference

Route workflow execution along different paths based on node outputs.

## Basic Routing

A node outputs a `route` value; edges with matching routes are followed:

```python
from google.adk.workflow import Workflow
from google.adk.events.event import Event

def classify(node_input: str):
  if "error" in node_input:
    return Event(data=node_input, route="error")
  return Event(data=node_input, route="success")

def handle_success(node_input: str) -> str:
  return f"Success: {node_input}"

def handle_error(node_input: str) -> str:
  return f"Error: {node_input}"

agent = Workflow(
    name="router",
    edges=[
        ('START', classify),
        (classify, handle_success, "success"),
        (classify, handle_error, "error"),
    ],
)
```

## Route Value Types

Routes can be `str`, `bool`, or `int`:

```python
# String routes
(decision_node, path_a, "approve")
(decision_node, path_b, "reject")

# Boolean routes
(decision_node, yes_path, True)
(decision_node, no_path, False)

# Integer routes
(decision_node, path_0, 0)
(decision_node, path_1, 1)
```

## Default Route

Use `'__DEFAULT__'` as a fallback when no other route matches:

```python
edges=[
    ('START', classify),
    (classify, handle_success, "success"),
    (classify, handle_error, "error"),
    (classify, handle_unknown, '__DEFAULT__'),  # Fallback
]
```

Only one default route per node is allowed.

**No duplicate edges:** Two edges from the same source to the same target are rejected, even with different routes. If you need both a named route and `__DEFAULT__` to reach the same destination, use a thin wrapper function for the default path.

## Dynamic Routing with Functions

Use a callable `route` for runtime decisions:

```python
from google.adk.agents.context import Context

def router(ctx: Context, node_input: dict):
  score = node_input.get("score", 0)
  if score > 0.8:
    return Event(data=node_input, route="high")
  elif score > 0.5:
    return Event(data=node_input, route="medium")
  else:
    return Event(data=node_input, route="low")
```

## Multi-Route (Fan-Out)

A node can output multiple routes to trigger multiple downstream paths:

```python
def fan_out_router(node_input: str):
  return Event(data=node_input, route=["path_a", "path_b"])

agent = Workflow(
    name="multi_route",
    edges=[
        ('START', fan_out_router),
        (fan_out_router, branch_a, "path_a"),
        (fan_out_router, branch_b, "path_b"),
    ],
)
```

## List of Routes on a Single Edge

An edge can match multiple routes by passing a list as the route value. The edge fires if the node output matches **any** route in the list:

```python
agent = Workflow(
    name="multi_match",
    edges=[
        ('START', classifier),
        (classifier, handler_a, ["route_x", "route_y"]),  # fires on either
        (classifier, handler_b, "route_z"),
    ],
)
```

This is useful when multiple route values should lead to the same downstream node without duplicating edges.

## Looping

Create loops with conditional exit by routing back to an earlier node:

```python
def increment(ctx: Context, node_input: str):
  count = ctx.state.get("count", 0) + 1
  ctx.state["count"] = count
  if count >= 3:
    return Event(data=f"Done after {count} iterations", route="exit")
  return Event(data=f"Iteration {count}", route="continue")

def process(node_input: str) -> str:
  return f"Processing: {node_input}"

def finish(node_input: str) -> str:
  return f"Finished: {node_input}"

agent = Workflow(
    name="loop",
    edges=[
        ('START', process),
        (process, increment),
        (increment, process, "continue"),  # Loop back
        (increment, finish, "exit"),       # Exit loop
    ],
)
```

**Important**: Cycles must have at least one routed edge (unconditional cycles are rejected during graph validation).

## Unconditional Edges

Edges without a route value are unconditional — they always fire:

```python
edges=[
    ('START', node_a),       # Unconditional
    (node_a, node_b),        # Unconditional (always fires)
]
```

If a node has both routed and unconditional edges, the unconditional edges fire only when no route is set on the output event.
