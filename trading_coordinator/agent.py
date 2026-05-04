import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ==============================================================================
# [ADK 2.0 FEATURE]: Canonical Primitives
# In ADK 2.0, the core declarative intelligent block is the `Agent` imported directly
# from `google.adk.agents`. The `Workflow` class provides pure graph orchestration.
# ==============================================================================
from google.adk.agents import Agent
from google.adk import Context
from google.adk.workflow import Workflow, JoinNode
from google.adk.workflow import node
from google.adk.events.event import Event
from google.adk.apps.app import App, ResumabilityConfig
from google.genai import types

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
    mode="single_turn",
    instruction="""
    You are a friendly and helpful cryptocurrency trading assistant.
    The user asked something outside of standard trading operations.
    Kindly greet them and provide a very clear markdown list of examples of what they CAN ask:
    1. Profile Inquiry (KYC, balances)
    2. Spot Trading (Buying/Selling)
    3. Derivative Trading (Perpetuals, leverage)
    4. External Transfers
    5. Portfolio Summary (parallel profile + market data)
    """,
)

# ==============================================================================
# [ADK 2.0 FEATURE]: Direct LLM Classification (Zero Hardcoded Routers)
# ==============================================================================
_intent_classifier_llm = Agent(
    name="intent_classifier_llm",
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=20,
    ),
    instruction="""
    You are a master cryptocurrency trading intent classifier.
    Your strict job is to read the user's input and return ONLY ONE of the exact route names below:

    - "profile": if the user asks about checking their account, balances, or KYC status.
    - "spot": if the user wants to buy or sell cryptocurrency on the spot market.
    - "derivative": if the user wants to trade perpetual futures, use leverage, or open positions.
    - "transfer": if the user wants to send, transfer, or withdraw cryptocurrency.
    - "portfolio": if the user asks for a portfolio summary, market overview, or combined account + market view.

    If none of the above apply, return exactly "__DEFAULT__".
    Do not add any markdown formatting, explanation, spaces, or punctuation. Just the exact lowercase word.
    """,
)

@node(name="intent_classifier", rerun_on_resume=True)
async def intent_classifier(ctx: Context, node_input) -> Event:
    """Runs the LLM classifier and emits a route for graph dictionary routing."""
    result = await ctx.run_node(_intent_classifier_llm, node_input)
    route = str(result).strip().strip('"').strip("'")
    return Event(route=route, output=node_input)

# ==============================================================================
# [ADK 2.0 FEATURE]: @node Function Nodes for Parallel Fan-Out
# These are lightweight code-only nodes used in the JoinNode parallel pattern.
# ==============================================================================
@node(name="portfolio_profile_node")
def portfolio_profile_node(node_input) -> Event:
    """Fetches user profile data for the portfolio fan-out branch."""
    from shared.tools import get_user_profile
    return Event(output=get_user_profile())

@node(name="market_data_node")
def market_data_node(node_input) -> Event:
    """Fetches simulated real-time market data for key crypto assets."""
    market_prices = {
        "BTC/USDT": {"price": 95420.50, "change_24h": "+2.3%"},
        "ETH/USDT": {"price": 3512.80, "change_24h": "-0.8%"},
        "SOL/USDT": {"price": 178.25, "change_24h": "+5.1%"},
    }
    return Event(output=market_prices)

# ==============================================================================
# [ADK 2.0 FEATURE]: JoinNode for Fan-Out / Fan-In Parallel Execution
# The portfolio_join node waits for BOTH profile_agent and market_data_node to
# complete, then assembles their combined outputs for the summary agent.
# ==============================================================================
portfolio_join = JoinNode(name="portfolio_join")

portfolio_summary = Agent(
    name="portfolio_summary",
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    instruction="""
    You are a portfolio summary agent. You receive combined data from two parallel sources:
    1. User profile data (balances, KYC status)
    2. Real-time market data (current prices and 24h changes)

    Synthesize both into a clear, professional portfolio report showing:
    - Account status and balances
    - Current market prices for held assets
    - Estimated portfolio value based on current prices
    """,
)

# ==============================================================================
# [ADK 2.0 FEATURE]: Dictionary Routing Map + JoinNode Parallel Edges
# ==============================================================================
from tools.compliance import compliance_check_tool

root_agent = Workflow(
    name="trading_coordinator",
    edges=[
        ("START", compliance_check_tool),
        (compliance_check_tool, {"pass": intent_classifier}),
        (intent_classifier, {
            "profile": profile_agent,
            "spot": spot_agent,
            "derivative": derivative_agent,
            "transfer": transfer_agent,
            "portfolio": portfolio_profile_node,
            "__DEFAULT__": default_handler,
        }),
        (intent_classifier, {"portfolio": market_data_node}),
        (portfolio_profile_node, portfolio_join),
        (market_data_node, portfolio_join),
        (portfolio_join, portfolio_summary),
    ]
)


# ==============================================================================
# [ADK 2.0 FEATURE]: App Packaging & State Resumability
# ==============================================================================
app = App(
    name="trading_coordinator",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

agent = root_agent
