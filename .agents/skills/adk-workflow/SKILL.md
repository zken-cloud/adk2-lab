---
name: adk-workflow
description: This skill should be used when the user asks to "build an agent", "create an ADK agent", "build a workflow agent", "create a workflow", "define a graph", "add nodes and edges", "implement conditional routing", "add human-in-the-loop", "use parallel workers", "wrap an LLM agent in a workflow", "create a fan-out fan-in pattern", "add retry logic to a node", "test a workflow agent", "set up an agent project", "configure API keys for ADK", "use task mode", "delegate tasks", "create a single-turn agent", "use request_task", "use finish_task", "add tools to an agent", "use MCP tools", "use session state", "add callbacks", "create a multi-agent system", or mentions Workflow, FunctionNode, LlmAgentNode, JoinNode, ParallelWorker, LlmAgent, BaseLlmAgent, Edge, RequestTaskTool, FinishTaskTool, mode='task', or mode='single_turn'. Provides comprehensive guidance for building ADK agents, from basic LLM agents with tools to graph-based workflow agents and task delegation.
---

# ADK Agent Development

## Getting Started

For environment setup, API key configuration, basic LLM agent creation, tool definitions, running agents, and sample projects, consult **`references/getting-started.md`**.

Quick setup:

```bash
pip install google-adk        # Install
adk create my_agent           # Scaffold project
# Edit my_agent/.env with GOOGLE_API_KEY=...
# Edit my_agent/agent.py with agent definition
adk web my_agent/             # Run web UI at localhost:8000
```

Agent directory structure (required for CLI discovery):

```
my_agent/
├── __init__.py    # from . import agent
├── agent.py       # Must define root_agent
└── .env           # GOOGLE_API_KEY=... (not committed to git)
```

## Basic LLM Agent with Tools

```python
from google.adk.agents.llm_agent import Agent

def get_weather(city: str) -> dict:
  """Returns current weather for a city."""
  return {"city": city, "weather": "sunny", "temp": "72F"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="You are a helpful assistant. Use get_weather for weather queries.",
    tools=[get_weather],
)
```

Tools are Python functions -- the name, docstring, and type hints become the tool schema the LLM sees. For all tool types (MCP, OpenAPI, Google API, built-in, BaseTool, BaseToolset), consult **`references/tool-catalog.md`**.

## Agent Modes: Chat, Task, and Single-Turn

Agents support three delegation modes via the `mode` parameter:

| Mode | Delegation Tool | User Interaction | Use Case |
|------|----------------|------------------|----------|
| `chat` (default) | `transfer_to_agent` | Full chat | General assistants |
| `task` | `request_task_{name}` | Multi-turn (can chat) | Structured I/O tasks |
| `single_turn` | `request_task_{name}` | None | Autonomous tasks |

### Task Mode (Structured Delegation)

```python
from google.adk.workflow.agents.base_llm_agent import BaseLlmAgent
from google.adk.workflow.agents.llm_agent import LlmAgent
from pydantic import BaseModel

class ResearchInput(BaseModel):
  topic: str
  depth: str = 'standard'

class ResearchOutput(BaseModel):
  summary: str
  key_findings: str

researcher = BaseLlmAgent(
    name='researcher',
    mode='task',
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    instruction='Research the topic, then call finish_task with results.',
    description='Researches topics.',
    tools=[search_web],
)

root_agent = LlmAgent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[researcher],
    instruction='Delegate research to the researcher using request_task_researcher.',
)
```

### Single-Turn Mode (Autonomous)

```python
summarizer = BaseLlmAgent(
    name='summarizer',
    mode='single_turn',
    output_schema=SummaryOutput,
    instruction='Summarize the content and call finish_task. No user interaction.',
    description='Summarizes documents autonomously.',
    tools=[extract_text],
)
```

For full task mode details, schemas, mixed-mode patterns, and the delegation lifecycle, consult **`references/task-mode.md`**.

## Workflow Agents

A `Workflow` extends the basic agent with graph-based execution. Instead of a single LLM deciding what to do, define explicit nodes and edges:

```python
from google.adk.workflow import Workflow

def greet(node_input: str) -> str:
  return f"Hello, {node_input}!"

root_agent = Workflow(
    name="greeter",
    edges=[('START', greet)],
)
```

### Core Concepts

A workflow has three building blocks:

1. **Nodes** -- units of work (functions, LLM agents, tools)
2. **Edges** -- connections between nodes, optionally with route conditions
3. **START** -- the built-in entry point that receives user input

## Node Types

Any "NodeLike" is accepted in edges and auto-wrapped:

| Python Object | Wrapped As | Default rerun_on_resume |
|--------------|-----------|------------------------|
| Function/callable | `FunctionNode` | `False` |
| `LlmAgent` | `LlmAgentNode` | `True` |
| Other `BaseAgent` | `AgentNode` | `False` |
| `BaseTool` | `ToolNode` | `False` |
| `BaseNode` subclass | Used as-is | Per subclass |

## Function Nodes

Functions are the most common node type. Parameter resolution:

| Parameter | Source |
|-----------|--------|
| `ctx` | Workflow `Context` object |
| `node_input` | Output from predecessor node |
| Any other name | `ctx.state[param_name]` |

```python
from google.adk.agents.context import Context

def process(ctx: Context, node_input: Any, user_name: str) -> str:
  # node_input = predecessor output; user_name = ctx.state['user_name']
  # NOTE: START node outputs types.Content (not str) unless input_schema is set
  return f"{user_name}: {node_input}"
```

Return `None` to suppress downstream triggering. Return an `Event` for routing or state updates:

```python
from google.adk.events.event import Event

def classify(node_input: str):
  if "urgent" in node_input:
    return Event(data=node_input, route="urgent")
  return Event(data=node_input, route="normal", state={"processed": True})
```

## Edge Patterns

```python
# Sequential chain
edges = [('START', a), (a, b), (b, c)]

# Or use Edge.chain
from google.adk.workflow import Edge
edges = Edge.chain('START', a, b, c)

# Conditional routing
edges = [
    ('START', classifier),
    (classifier, success_handler, "success"),
    (classifier, error_handler, "error"),
    (classifier, fallback_handler, '__DEFAULT__'),  # Fallback route
]

# Fan-out (parallel branches)
edges = [('START', (branch_a, branch_b, branch_c))]

# Fan-in with JoinNode
from google.adk.workflow.join_node import JoinNode
join = JoinNode(name="merge")
edges = [((branch_a, branch_b), join), (join, final)]
# JoinNode output: {"branch_a": output_a, "branch_b": output_b}

# Looping (must have at least one routed edge)
edges = [
    ('START', process),
    (process, check),
    (check, process, "continue"),
    (check, finish, "exit"),
]
```

## LLM Agent Nodes

Use `google.adk.agents.llm_agent.LlmAgent` in workflow edges. It is auto-wrapped as `LlmAgentNode`, which emits `Event(data=...)` for downstream data passing:

```python
from google.adk.agents.llm_agent import LlmAgent
from pydantic import BaseModel

class DraftOutput(BaseModel):
  title: str
  content: str

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a draft based on the user's request.",
    output_schema=DraftOutput,  # Always set for structured output
    output_key="draft",         # Also store in state['draft']
)

agent = Workflow(
    name="pipeline",
    edges=[('START', writer), (writer, process_draft)],
)
```

**Always use `output_schema`** (Pydantic model) on LLM agents in workflows. Without it, the output is `types.Content` which causes type errors in downstream function nodes and serialization failures with JoinNode/database sessions.

**Do NOT use `google.adk.workflow.agents.llm_agent.LlmAgent`** as an intermediate node — it is mesh-based and does not emit data output events. Use it only as a top-level coordinator or nested workflow root.

Use `single_turn=True` on `LlmAgentNode` to isolate from session history. For tools, callbacks, output schemas, and advanced LLM configuration, consult **`references/llm-agent-nodes.md`**.

## Parallel Processing

Process list items concurrently with `ParallelWorker`:

```python
from google.adk.workflow.parallel_worker import ParallelWorker
from google.adk.workflow.node import node

@node(parallel_worker=True)
def process_item(node_input: int) -> int:
  return node_input * 2

# Input: [1, 2, 3] -> Output: [2, 4, 6]
```

## Human-in-the-Loop

Pause execution and request user input:

```python
from google.adk.events.request_input import RequestInput

async def approval_gate(ctx: Context, node_input: str):
  yield RequestInput(
      message="Approve this action?",
      response_schema={"type": "string"},
  )
```

HITL works in two modes:

**Resumable mode** (recommended for multi-step HITL): Export an `App` with resumability. The workflow checkpoints state and resumes at the interrupted node.

```python
from google.adk.apps.app import App, ResumabilityConfig

app = App(
    name="my_app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

**Non-resumable mode** (simpler, no App needed): The workflow replays from START on each user response, reconstructing state from session events. Works automatically for simple HITL but replays all nodes up to the interrupt point.

When `rerun_on_resume=False` (default for FunctionNode), the user's response becomes the node's output. When `rerun_on_resume=True`, the node reruns with `ctx.resume_inputs` populated. For details, consult **`references/human-in-the-loop.md`**.

## Retry Configuration

```python
from google.adk.workflow.retry_config import RetryConfig
from google.adk.workflow import FunctionNode

node = FunctionNode(
    flaky_call,
    retry_config=RetryConfig(max_attempts=3, initial_delay=1.0, backoff_factor=2.0),
)
```

## Agent Directory Convention

For CLI discovery (`adk web`, `adk run`):

```
my_workflow/
  __init__.py    # from . import agent
  agent.py       # root_agent = Workflow(...)
```

## Best Practices (MUST FOLLOW)

### Use Pydantic Models, Not Raw Dicts

**Always define Pydantic `BaseModel` classes** for function node inputs, outputs, LLM `output_schema`, and structured data. Never use `dict[str, Any]` when the shape is known:

```python
# ❌ WRONG: raw dicts
def lookup_flights(node_input: dict[str, Any]) -> dict[str, Any]:
  return {"flight_cost": 500, "details": "Economy"}

# ✅ CORRECT: typed schemas
class FlightInfo(BaseModel):
  flight_cost: int
  details: str

def lookup_flights(node_input: Itinerary) -> FlightInfo:
  return FlightInfo(flight_cost=500, details="Economy")
```

This applies to ALL data flowing through the graph: node inputs, node outputs, JoinNode results, LLM output schemas, and HITL response schemas.

### Emit Content Events for Web UI Display

`event.data` is internal — only `event.content` renders in the ADK web UI. For user-visible output, yield a content event:

```python
from google.genai import types

def final_output(node_input: str):
  yield Event(
      content=types.Content(role='model', parts=[types.Part.from_text(node_input)]),
  )
  yield Event(data=node_input)
```

LLM agents emit content events automatically. Add them explicitly for function nodes that produce user-facing results.

### Use Core LlmAgent (auto-wrapped as LlmAgentNode)

Use `google.adk.agents.llm_agent.LlmAgent` in workflow edges. It is auto-wrapped as `LlmAgentNode`, which emits `Event(data=...)` for downstream data passing:

```python
from google.adk.agents.llm_agent import LlmAgent

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a draft.",
    output_schema=DraftOutput,  # Always set for structured output
)

# LlmAgent is auto-wrapped as LlmAgentNode in edges
agent = Workflow(name="pipeline", edges=[('START', writer), (writer, next_step)])
```

**Note:** `google.adk.workflow.agents.llm_agent.LlmAgent` is a mesh-based agent that does NOT emit data output events. It only passes data via `output_key` → state. Use it only as a top-level coordinator or nested workflow root, not as an intermediate node that needs to pass data to downstream function nodes.

### Set State via Event, Not ctx.state

**Prefer `Event(state=...)` over `ctx.state[key] = ...`** for writing state. Event-based state is persisted in event history and replayable during non-resumable HITL. Direct `ctx.state` mutations are side effects that may be lost on replay.

```python
# ✅ Preferred
def save(node_input: str):
  return Event(data=node_input, state={"user_request": node_input})

# ❌ Avoid
def save(ctx: Context, node_input: str) -> str:
  ctx.state["user_request"] = node_input
  return node_input
```

### Workflow Data Rules

- **`Event.data` must be JSON-serializable.** FunctionNode auto-converts BaseModel returns via `model_dump()`. Never store `types.Content` or other non-serializable objects in `Event.data`.
- **`output_key` stores dicts, not BaseModel instances.** LLM agents with `output_schema` run `validate_schema()` → `model_dump()`, so `ctx.state[output_key]` is a plain dict.
- **`ctx.state.get(key)` returns a dict.** Use dict access (`data["field"]`) or reconstruct (`MyModel(**data)`) for typed access.

## Additional Resources

### Reference Files

For detailed patterns and techniques, consult:

- **`references/getting-started.md`** -- Environment setup, API keys, basic LLM agents with tools, running agents, complete sample projects
- **`references/tool-catalog.md`** -- All tool types: function tools, MCP, OpenAPI, Google API, built-in tools, BaseTool, BaseToolset, ToolContext, LongRunningFunctionTool
- **`references/task-mode.md`** -- Task delegation: mode='task', mode='single_turn', input/output schemas, request_task, finish_task, mixed-mode patterns
- **`references/multi-agent.md`** -- Multi-agent patterns: chat transfer, SequentialAgent, ParallelAgent, LoopAgent, model configuration
- **`references/session-and-state.md`** -- Session state, artifacts, memory services, state key conventions
- **`references/callbacks-and-plugins.md`** -- All callback types, signatures, plugin system, built-in plugins
- **`references/function-nodes.md`** -- FunctionNode details, @node decorator, generators, auto type conversion
- **`references/routing-and-conditions.md`** -- Conditional branching, dynamic routing, loops, multi-route fan-out
- **`references/state-and-events.md`** -- Context API, shared state, Event fields, intermediate content
- **`references/llm-agent-nodes.md`** -- LlmAgentNode, instructions, tools, all callback types, output schemas
- **`references/human-in-the-loop.md`** -- RequestInput, resume behavior, multi-step HITL, resumability config
- **`references/parallel-and-fanout.md`** -- ParallelWorker, JoinNode, fan-out/fan-in, diamond pattern, SequentialAgent/ParallelAgent
- **`references/advanced-patterns.md`** -- Nested workflows, dynamic nodes, retry config, custom BaseNode, ToolNode, AgentNode, graph validation
- **`references/testing.md`** -- pytest patterns, MockModel, InMemoryRunner, testing utilities
- **`references/import-paths.md`** -- Quick-reference import table for all ADK components
