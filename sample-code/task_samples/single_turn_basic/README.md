# Single-Turn Basic

Demonstrates the single-turn delegation pattern with a coordinator and
an autonomous summarizer child.

## Flow

1. User asks the coordinator to summarize something.
2. Coordinator calls `request_task_summarizer(...)` to delegate.
3. Summarizer works autonomously (no user interaction).
4. Summarizer calls `finish_task` with the result.
5. Coordinator presents the summary to the user.

Single-turn agents never interact with the user directly. They receive
input, do their work, and return a result.

## Run

```bash
adk web contributing/task_samples/
```

Select **single_turn_basic** in the web UI.

## Testing Prompts

- Summarize the history of the internet
- Give me a summary of machine learning
- Summarize https://example.com/article
- Can you summarize the key developments in space exploration?

**Test parallel single_turn delegation**

- Summarize ADK and wikepedia separately.
