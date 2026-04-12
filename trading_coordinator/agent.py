import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ==============================================================================
# [ADK 2.0 FEATURE]: Canonical Primitives
# In ADK 2.0, the core declarative intelligent block is the `Agent` imported directly
# from `google.adk.agents`. The `Workflow` class provides pure graph orchestration.
# ==============================================================================
from google.adk.agents import Agent
from google.adk.workflow import Workflow
from google.adk.apps.app import App, ResumabilityConfig

from agents.profile import profile_agent
from agents.spot import spot_agent
from agents.derivative import derivative_agent
from agents.transfer import transfer_agent

# ==============================================================================
# [ADK 2.0 FEATURE]: Declarative Fallback Agents
# Instead of hardcoding a Python function string to handle unrouted user inputs, 
# we define a fully declarative LLM `Agent` node to handle the '__DEFAULT__' route natively.
# ==============================================================================
default_handler = Agent(
    name="default_handler",
    model="gemini-3.1-flash-lite-preview",
    instruction="""
    You are a friendly and helpful cryptocurrency trading assistant.
    The user asked something outside of standard trading operations.
    Kindly greet them and provide a very clear markdown list of examples of what they CAN ask:
    1. Profile Inquiry (KYC, balances)
    2. Spot Trading (Buying/Selling)
    3. Derivative Trading (Perpetuals, leverage)
    4. External Transfers
    """,
)

# ==============================================================================
# [ADK 2.0 FEATURE]: Direct LLM Classification (Zero Hardcoded Routers)
# We eliminate intermediate Python routing functions entirely. This `Agent` reads
# the user intent and outputs exactly one lowercase routing key directly.
# ==============================================================================
intent_classifier = Agent(
    name="intent_classifier",
    model="gemini-3.1-flash-lite-preview",
    instruction="""
    You are a master cryptocurrency trading intent classifier.
    Your strict job is to read the user's input and return ONLY ONE of the exact route names below:
    
    - "profile": if the user asks about checking their account, balances, or KYC status.
    - "spot": if the user wants to buy or sell cryptocurrency on the spot market.
    - "derivative": if the user wants to trade perpetual futures, use leverage, or open positions.
    - "transfer": if the user wants to send, transfer, or withdraw cryptocurrency.
    
    If none of the above apply, return exactly "__DEFAULT__".
    Do not add any markdown formatting, explanation, spaces, or punctuation. Just the exact lowercase word.
    """,
)

# ==============================================================================
# [ADK 2.0 FEATURE]: Dictionary Routing Map (Pure Conditional Edge Flow)
# This is the canonical ADK 2.0 Fan-out routing syntax. A dictionary map attached
# to a source node automatically directs execution to the matching target node based
# on the raw output of the source node.
# ==============================================================================
root_agent = Workflow(
    name="trading_coordinator",
    edges=[
        ("START", intent_classifier),
        (intent_classifier, {
            "profile": profile_agent,
            "spot": spot_agent,
            "derivative": derivative_agent,
            "transfer": transfer_agent,
            "__DEFAULT__": default_handler,
        }),
    ]
)

# ==============================================================================
# [ADK 2.0 FEATURE]: App Packaging & State Resumability
# The `App` container bundles the root workflow and enables persistent state resumability 
# to support advanced features like long-running tasks and human-in-the-loop interruptions.
# ==============================================================================
app = App(
    name="trading_coordinator",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

agent = root_agent
