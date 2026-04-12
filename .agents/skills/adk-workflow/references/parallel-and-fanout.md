# Parallel Execution, Fan-Out, and Fan-In Reference

Execute multiple nodes concurrently and collect their results.

## Imports

```python
from google.adk.workflow import Workflow
from google.adk.workflow.parallel_worker import ParallelWorker
from google.adk.workflow.join_node import JoinNode
from google.adk.workflow.node import node
```

## Fan-Out: Multiple Branches

Send output to multiple nodes simultaneously using tuple syntax:

```python
def analyze_text(node_input: str) -> str:
  return f"Analysis: {node_input}"

def translate_text(node_input: str) -> str:
  return f"Translation: {node_input}"

def summarize_text(node_input: str) -> str:
  return f"Summary: {node_input}"

agent = Workflow(
    name="fan_out",
    edges=[
        ('START', (analyze_text, translate_text, summarize_text)),
    ],
)
```

Each branch receives the same input and runs concurrently.

## Fan-In: JoinNode

Collect outputs from multiple branches before continuing:

```python
join = JoinNode(name="collect_results")

agent = Workflow(
    name="fan_out_fan_in",
    edges=[
        ('START', (analyze_text, translate_text, summarize_text)),
        ((analyze_text, translate_text, summarize_text), join),
        (join, final_processor),
    ],
)
```

### JoinNode Output Format

JoinNode outputs a dictionary mapping predecessor names to their outputs:

```python
# JoinNode output:
# {
#   "analyze_text": "Analysis: hello",
#   "translate_text": "Translation: hello",
#   "summarize_text": "Summary: hello",
# }

def final_processor(node_input: dict) -> str:
  analysis = node_input["analyze_text"]
  translation = node_input["translate_text"]
  summary = node_input["summarize_text"]
  return f"Combined: {analysis}, {translation}, {summary}"
```

### JoinNode Behavior

- Waits for **all** predecessor nodes to complete
- Emits `no_trigger=True` events while still waiting
- Only triggers downstream when all inputs are received
- Stores partial inputs in workflow state

**Serialization warning:** JoinNode stores partial inputs in session state while waiting. If predecessors are LLM agents without `output_schema`, the stored values are `types.Content` objects which are **not JSON-serializable**. This causes `TypeError` with SQLite/database session services. Fix: use `output_schema` on LLM agents feeding into a JoinNode.

## ParallelWorker: Process Lists in Parallel

Apply the same node to each item in a list concurrently:

```python
def process_item(node_input: int) -> int:
  return node_input * 2

parallel = ParallelWorker(node(process_item))

def produce_list(node_input: str) -> list:
  return [1, 2, 3, 4, 5]

agent = Workflow(
    name="parallel_processing",
    edges=[
        ('START', produce_list),
        (produce_list, parallel),
    ],
)
# Output: [2, 4, 6, 8, 10]
```

### ParallelWorker Details

- Input: a **list** (or single item, which gets wrapped in a list)
- Output: a **list** of results in the same order as inputs
- Empty list input produces empty list output
- Each item is processed by a dynamically created worker node
- Workers are named `{parent_name}@{index}` (e.g., `process_item@0`)
- Default `rerun_on_resume=True`

### ParallelWorker with @node Decorator

```python
@node(parallel_worker=True)
def process_item(node_input: int) -> int:
  return node_input * 2

# Equivalent to: ParallelWorker(FunctionNode(process_item_fn))
```

### ParallelWorker with LLM Agents

```python
analyzer = LlmAgent(
    name="analyzer",
    model="gemini-2.5-flash",
    instruction="Analyze this data point and provide insights.",
)

parallel_analyzer = ParallelWorker(LlmAgentNode(agent=analyzer, single_turn=True))

agent = Workflow(
    name="parallel_analysis",
    edges=[
        ('START', split_data),
        (split_data, parallel_analyzer),
        (parallel_analyzer, aggregate_results),
    ],
)
```

## Diamond Pattern

Fan-out then fan-in (diamond shape):

```python
def splitter(node_input: str) -> str:
  return node_input

def branch_a(node_input: str) -> str:
  return f"A: {node_input}"

def branch_b(node_input: str) -> str:
  return f"B: {node_input}"

join = JoinNode(name="merge")

def combiner(node_input: dict) -> str:
  return f"Combined: {node_input['branch_a']} + {node_input['branch_b']}"

agent = Workflow(
    name="diamond",
    edges=[
        ('START', splitter),
        (splitter, (branch_a, branch_b)),
        ((branch_a, branch_b), join),
        (join, combiner),
    ],
)
```

## SequentialAgent and ParallelAgent

Convenience subclasses for common patterns:

```python
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent

# Sequential: runs sub_agents in order
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[writer_agent, reviewer_agent, editor_agent],
)
# Equivalent to: START -> writer -> reviewer -> editor

# Parallel: runs sub_agents concurrently
parallel = ParallelAgent(
    name="concurrent",
    sub_agents=[analyzer_agent, translator_agent, summarizer_agent],
)
# Equivalent to: START -> (analyzer, translator, summarizer)
```
