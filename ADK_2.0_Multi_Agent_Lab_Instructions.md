# 🎓 ADK 2.0: Architecting Multi-Agent Systems (1-Hour Lab)

**Welcome to the official Google Agent Development Kit (ADK) 2.0 Masterclass!** 

In this immersive 60-minute hands-on workshop, you will explore the three primary orchestration patterns available in ADK 2.0, compare their execution profiles, and execute them using the native CLI runtime.

**📚 Official Reference:** [ADK 2.0 Documentation Portal](https://adk.dev/2.0/)

---

## ⏱️ 1-Hour Workshop Syllabus Breakdown

*   **Setup (5 mins):** Environment Configuration & Installation
*   **Module 1 (15 mins):** Graph-Based Workflows (Deterministic Control)
*   **Module 2 (20 mins):** Dynamic Delegation (Type-Safe Task Execution)
*   **Module 3 (20 mins):** Collaborative Mixed-Mode (Autonomous Orchestration)
*   **Module 4 (5 mins):** Architecture Selection & Q&A

---

## 🛠️ Setup & Installation: Step-by-Step Guide (5 Mins)

Before diving into the modules, please ensure your local environment is correctly configured with the latest ADK 2.0 framework.

### Step 1: Verify Prerequisites
Ensure your environment has the required tools installed and authenticated:
*   **Python 3.10+** with `uv` installed.
*   **GCP Access:** Authenticated via Application Default Credentials.
    ```bash
    gcloud auth application-default login
    ```

### Step 2: Download the ADK 2.0 Package
Download the latest wheel file from the official GCS bucket to your project root:
```bash
gsutil cp 'gs://agent_framework/2.0/latest/google_adk-*.whl' .
```

### Step 3: Initialize the Virtual Environment
Set up and activate a clean Python virtual environment using `uv`:
```bash
# Create the virtual environment
uv venv --python "python3.11" ".venv"

# Activate the environment
source .venv/bin/activate
```

### Step 4: Install Dependencies
Install the ADK 2.0 package into your environment:
```bash
# Install using uv pip
uv pip install google_adk-*.whl

```

---

## 🏛️ Module 1: Declarative Graph-Based Workflows (15 Mins)

**📂 Directory:** `trading_coordinator`

### 📖 Core Concepts
This architecture uses static, deterministic routing paths. The orchestrator acts as a dedicated intent classifier, returning a specific routing keyword (e.g., `"spot"` or `"transfer"`). The framework then uses a standard dictionary lookup to route the flow to the correct sub-agent.

### 💻 Implementation Snapshot
*   **Edge Mapping:** Defined explicitly in the workflow logic:
    ```python
    edges = [
        ("START", intent_classifier),
        (intent_classifier, {
            "profile": profile_agent,
            "spot": spot_agent,
            "__DEFAULT__": default_handler,
        })
    ]
    ```

### 🚀 Step-by-Step Execution
1.  **Launch the Agent:** Run the following command from your terminal to start the interactive CLI session:
    ```bash
    adk run trading_coordinator
    ```
2.  **Test the Workflow:** Try asking a query that maps to a specific intent, such as:
    > *"I want to check my user profile."*
3.  **Observe the Output:** Notice how the flow is deterministically routed to the `profile` agent based on the defined edges.

### ⚖️ Trade-offs
*   **✅ Deterministic Control:** Flow paths are explicitly defined and highly predictable.
*   **❌ Development Overhead:** Requires manual edge definitions for each sub-agent.

### 🎯 Enterprise Application Scenarios
Ideal for structured compliance pipelines, customer service IVR decision trees, and strict procedural workflows.

---

## ⚡ Module 2: Dynamic Task Delegation (20 Mins)

**📂 Directory:** `trading_agent_dynamic`

### 📖 Core Concepts
This design removes hardcoded edges entirely. Instead, sub-agents operate in `mode='task'` with typed inputs and outputs defined by Pydantic schemas. The main orchestrator discovers available operations dynamically and delegates tasks autonomously using auto-injected `request_task_{agent_name}` tools.

### 💻 Implementation Snapshot
*   **Schema Guardrails:** Built-in validation using typed data models:
    ```python
    class SpotTradeInput(BaseModel):
        symbol: str
        amount: float
        
    spot_agent = Agent(
        mode="task",
        input_schema=SpotTradeInput,
        output_schema=SpotTradeOutput,
    )
    ```

### 🚀 Step-by-Step Execution
1.  **Launch the Agent:** Run the following command to start the dynamic delegation session:
    ```bash
    adk run trading_agent_dynamic
    ```
2.  **Test Task Delegation:** Request an action that requires structured data input:
    > *"Buy 2.5 BTC at market price."*
3.  **Observe the Output:** Notice how the orchestrator autonomously invokes the `request_task_spot` tool, passing the validated `SpotTradeInput` schema to the task agent.

### ⚖️ Trade-offs
*   **✅ Type Safety:** Ensures structured data formatting and validation between operations.
*   **❌ Higher Token Utilization:** Requires the LLM to continually process validation schemas.

### 🎯 Enterprise Application Scenarios
Best suited for multi-step data transformation pipelines, dynamic document generation, and complex programmatic execution flows.

---

## 🤝 Module 3: Collaborative Mixed-Mode Chat (20 Mins)

**📂 Directory:** `trading_agent_collaborative`

### 📖 Core Concepts
This approach uses natural conversational hand-offs (`mode='chat'`) combined with a highly flexible **Mixed-Mode** orchestration model:

*   `mode='single_turn'`: Ideal for fast autonomous lookups.
*   `mode='task'`: Useful for structured execution tasks.
*   `mode='chat'`: Perfect for conversational user interactions.

### 💻 Implementation Snapshot
*   **Unified Primitives:** Uses standard `Agent` blocks with specialized operational modes:
    ```python
    profile_agent = Agent(mode="single_turn")
    derivative_agent = Agent(mode="task")
    spot_agent = Agent(mode="chat")
    ```

### 🚀 Step-by-Step Execution
1.  **Launch the Agent:** Run the collaborative mixed-mode agent:
    ```bash
    adk run trading_agent_collaborative
    ```
2.  **Test Conversational Handoff:** Initiate a multi-turn conversation:
    > *"Can you help me with trading derivatives?"*
3.  **Observe the Output:** Notice how the orchestrator gracefully hands over control to the `derivative_agent`, which can then maintain its own conversational context or perform tasks.

### ⚖️ Trade-offs
*   **✅ Autonomous Routing:** Sub-agents hand off traffic seamlessly based on agent descriptions.
*   **❌ LLM Dependency:** Highly reliant on the underlying reasoning model to handle intent routing properly.

### 🎯 Enterprise Application Scenarios
Excellent for multi-departmental virtual assistants, integrated internal advisory portals, and conversational customer support systems.

