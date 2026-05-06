import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_derivative_trade

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    description="Handles perpetual futures, margin leverage trading, and opening derivative positions via structured execution.",
    instruction="""
    You are a structured task execution agent.
    Execute the requested leveraged derivative order and securely hand the execution receipt back to the parent.
    """,
    tools=[execute_derivative_trade],
)
