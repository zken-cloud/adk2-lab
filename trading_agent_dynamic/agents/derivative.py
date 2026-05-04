import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_derivative_trade
from shared.schemas import DerivativeTradeInput, DerivativeTradeOutput

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    input_schema=DerivativeTradeInput,
    output_schema=DerivativeTradeOutput,
    description="Handles perpetual futures, leverage trading, and opening derivative positions.",
    instruction="""
    You are a specialized derivatives execution agent.
    Read the parameters from your input schema, invoke `execute_derivative_trade`,
    and securely invoke finish_task to return the receipt to the coordinator.
    """,
    tools=[execute_derivative_trade],
)
