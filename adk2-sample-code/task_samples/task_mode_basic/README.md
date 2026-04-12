# Task Mode Basic

Demonstrates the full task delegation cycle with a coordinator and a
task-mode researcher child.

## Flow

1. User asks the coordinator to research a topic.
2. Coordinator calls `request_task_researcher(...)` to delegate.
3. Researcher uses `search_web` and `analyze_sources` tools.
4. Researcher can chat with the user for clarification.
5. Researcher calls `finish_task` when done.
6. Coordinator receives the result and responds to the user.

## Run

```bash
adk web contributing/task_samples/
```

Select **task_mode_basic** in the web UI.

## Testing Prompts

- Research quantum computing
- Can you research climate change in depth?
- What are the latest trends in AI?
- I need a detailed analysis of renewable energy
