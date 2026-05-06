import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_derivative_trade

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    description="Handles perpetual futures, leverage trading, and opening derivative positions.",
    instruction="""
    You are a derivatives and perpetual futures trading assistant.
    Review the approved derivatives order, confirm the leverage factor is safe (1x to 20x max),
    and invoke the `execute_derivative_trade` tool to open the position.
    Output the final position confirmation to the user.
    """,
    tools=[execute_derivative_trade],
)
