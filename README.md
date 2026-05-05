# ADK 2.0: Architecting Multi-Agent Systems (1-Hour Lab)

**Welcome to the official Google Agent Development Kit (ADK) 2.0 Masterclass!**

In this immersive 60-minute hands-on workshop, you will explore the three primary orchestration patterns available in ADK 2.0, compare their execution profiles, and execute them using the native CLI runtime.

**Official Reference:** [ADK 2.0 Documentation Portal](https://adk.dev/2.0/)

---

## Workshop Syllabus

*   **Setup (5 mins):** Environment Configuration & Installation
*   **Module 1 (15 mins):** Graph-Based Workflows (Deterministic Control)
*   **Module 2 (20 mins):** Dynamic Workflows with `@node` + `ctx.run_node()`
*   **Module 3 (20 mins):** Collaborative Mixed-Mode (Autonomous Orchestration)
*   **Wrap-up (5 mins):** Architecture Selection & Q&A

---

## Setup & Installation (5 Mins)

### Step 1: Verify Prerequisites
*   **Python 3.10+** with `uv` installed.
*   **GCP Access:** Authenticated via Application Default Credentials.
    ```bash
    gcloud auth application-default login
    ```

### Step 2: Initialize the Virtual Environment
```bash
uv venv --python "python3.11" ".venv"
source .venv/bin/activate
```

### Step 3: Install ADK 2.0
```bash
pip install google-adk==2.0.0b1
```

---

## Project Structure

```
adk2-lab/
├── shared/                          # Shared utilities across all modules
│   ├── tools.py                     # Mock data & tool functions (profile, spot, derivative, transfer, KYC)
│   └── schemas.py                   # Pydantic input/output schemas
├── trading_agent_graph/             # Module 1: Graph-Based Workflows
│   ├── agent.py                     # Workflow with dictionary routing, JoinNode, compliance
│   ├── agents/                      # Sub-agents (profile, spot, derivative, transfer)
│   └── tools/compliance.py          # @node KYC compliance check
├── trading_agent_dynamic/           # Module 2: Dynamic @node Workflows
│   ├── agent.py                     # Pure @node workflow with ctx.run_node() routing
│   ├── agents/                      # Sub-agent definitions (unused by @node approach)
│   └── tools/compliance.py          # @node compliance check
├── trading_agent_collaborative/     # Module 3: Collaborative Mixed-Mode
│   ├── agent.py                     # Root Agent with before_agent_callback + sub_agents
│   └── agents/                      # Sub-agents with mixed modes (single_turn, task, chat)
└── adk2-sample-code/                # ADK 2.0 reference samples
```

---

## Module 1: Graph-Based Workflows (15 Mins)

**Directory:** `trading_agent_graph/`

### Core Concepts
Static, deterministic routing via a `Workflow` with explicit edge definitions. An LLM-based intent classifier returns a routing keyword, and the framework uses a dictionary lookup to route to the correct sub-agent.

### Key ADK 2.0 Features Demonstrated
- **`Workflow` with dictionary routing** — deterministic edge-based control flow
- **`@node` intent classifier** — wraps an LLM `Agent` and emits `Event(route=...)` for graph routing
- **`JoinNode` fan-out/fan-in** — parallel execution of `portfolio_profile_node` and `market_data_node`, joined before a summary agent
- **`@node` compliance pre-processing** — KYC validation before intent classification
- **`generate_content_config`** — `temperature=0.0` on the classifier for deterministic output
- **Sub-agent `mode` settings** — `single_turn` for profile, `task` for spot/derivative/transfer
- **`App` with `ResumabilityConfig`** — state resumability support

### Implementation Highlights
```python
# Dictionary routing with JoinNode parallel fan-out
root_agent = Workflow(
    name="trading_agent_graph",
    edges=[
        ("START", compliance_check_tool),
        (compliance_check_tool, {"pass": intent_classifier}),
        (intent_classifier, {
            "profile": profile_agent,
            "spot": spot_agent,
            "portfolio": portfolio_profile_node,
            "__DEFAULT__": default_handler,
        }),
        (intent_classifier, {"portfolio": market_data_node}),
        (portfolio_profile_node, portfolio_join),
        (market_data_node, portfolio_join),
        (portfolio_join, portfolio_summary),
    ]
)
```

### Run It
```bash
.venv/bin/adk run trading_agent_graph
```

**Test prompts:**
- `"Check my profile"` — routes to profile agent
- `"Buy 2 BTC"` — routes to spot agent
- `"Show me my portfolio"` — triggers parallel fan-out (profile + market data), then summary
- `"What's the weather?"` — routes to default handler

### Trade-offs
*   **Deterministic control** — flow paths are explicit and predictable
*   **Development overhead** — requires manual edge definitions for each route

---

## Module 2: Dynamic Workflows with `@node` (20 Mins)

**Directory:** `trading_agent_dynamic/`

### Core Concepts
A pure `@node` workflow with no sub-agents. A single orchestrator `@node` function uses `ctx.run_node()` to programmatically invoke other `@node` functions based on LLM-parsed intent. This is the signature ADK 2.0 dynamic workflow pattern.

### Key ADK 2.0 Features Demonstrated
- **`@node` function nodes** — lightweight Python functions decorated with `@node`
- **`ctx.run_node()` dynamic dispatch** — programmatic node invocation at runtime
- **Direct `genai.Client()` LLM calls** — inline intent parsing without a full Agent
- **`ROUTE_MAP` dict routing** — maps parsed intents to `@node` functions
- **Inline KYC compliance** — warn-but-allow pattern via `check_kyc_status()`
- **`rerun_on_resume=True`** — ensures the orchestrator re-executes on state resume

### Implementation Highlights
```python
ROUTE_MAP = {
    "profile": profile_node,
    "spot": spot_node,
    "derivative": derivative_node,
    "transfer": transfer_node,
}

@node(rerun_on_resume=True)
async def dynamic_trading_workflow(ctx: Context, node_input: str):
    parsed = await ctx.run_node(parse_intent, node_input)
    target = ROUTE_MAP.get(parsed.get("intent"))
    if target:
        result = await ctx.run_node(target, parsed.get("params", {}))
```

### Run It
```bash
.venv/bin/adk run trading_agent_dynamic
```

**Test prompts:**
- `"Check my account"` — invokes profile_node
- `"Buy 2 BTC"` — invokes spot_node with parsed params
- `"Open 10x long ETH-PERP"` — invokes derivative_node
- `"Send 500 USDT to 0xabc123"` — invokes transfer_node

### Trade-offs
*   **Maximum flexibility** — routing logic is pure Python, not edge declarations
*   **Full programmatic control** — can add conditional logic, loops, error handling
*   **No LLM overhead for routing** — uses lightweight direct `genai` calls instead of Agent-based delegation

---

## Module 3: Collaborative Mixed-Mode (20 Mins)

**Directory:** `trading_agent_collaborative/`

### Core Concepts
Natural conversational hand-offs using a root `Agent` with specialized sub-agents in different operational modes. The LLM autonomously decides which sub-agent to delegate to based on agent descriptions.

### Key ADK 2.0 Features Demonstrated
- **Mixed-mode sub-agents** — `single_turn` (profile), `task` (derivative), `chat` (spot, transfer)
- **`before_agent_callback` compliance guardrail** — callback-based KYC check before each request
- **No `mode` on root agent** — per ADK 2.0 docs, `mode` is a subagent-only property
- **Pydantic `input_schema` / `output_schema`** — type-safe I/O on task-mode agents
- **Autonomous LLM routing** — the root agent uses sub-agent descriptions to delegate

### Implementation Highlights
```python
def compliance_guard(callback_context: CallbackContext) -> types.Content | None:
    kyc = check_kyc_status()
    if not kyc["verified"]:
        print(f"[Compliance Warning] {kyc['message']}")
    return None  # warn but allow

root_agent = Agent(
    name="trading_coordinator_collaborative",
    model="gemini-3.1-flash-lite-preview",
    before_agent_callback=compliance_guard,
    sub_agents=[profile_agent, spot_agent, derivative_agent, transfer_agent],
)
```

### Run It
```bash
.venv/bin/adk run trading_agent_collaborative
```

**Test prompts:**
- `"Check my profile"` — delegates to profile_agent (single_turn)
- `"Buy 1 ETH"` — delegates to spot_agent (chat mode, conversational)
- `"Open a 5x long ETH-PERP"` — delegates to derivative_agent (task mode, structured I/O)
- `"Send 100 USDT to 0xabc"` — delegates to transfer_agent (chat mode)

### Trade-offs
*   **Natural conversational flow** — sub-agents hand off seamlessly
*   **Minimal boilerplate** — no edge definitions or routing logic needed
*   **LLM dependency** — routing quality depends on the model's reasoning

---

## Shared Utilities

**Directory:** `shared/`

All three modules share centralized tool functions and Pydantic schemas:

- **`shared/tools.py`** — Mock user data (`MOCK_USERS`), `get_user_profile()`, `execute_spot_trade()`, `execute_derivative_trade()`, `execute_transfer()`, `check_kyc_status()`
- **`shared/schemas.py`** — `ProfileInput`, `ProfileOutput`, `SpotTradeInput`, `SpotTradeOutput`, `DerivativeTradeInput`, `DerivativeTradeOutput`, `TransferInput`, `TransferOutput`

---

## Architecture Comparison

| Dimension | Module 1: Graph | Module 2: Dynamic `@node` | Module 3: Collaborative |
|---|---|---|---|
| **Routing** | Dictionary edge map | `ctx.run_node()` dispatch | LLM-autonomous delegation |
| **Control** | Fully deterministic | Programmatic (Python logic) | LLM-driven |
| **Compliance** | `@node` pre-processing step | Inline KYC check | `before_agent_callback` |
| **Parallelism** | `JoinNode` fan-out/fan-in | Sequential (extensible) | N/A |
| **Type Safety** | Agent-level schemas | Direct tool calls | `input_schema`/`output_schema` |
| **Best For** | Regulated pipelines, IVR trees | Data pipelines, custom logic | Conversational assistants |
