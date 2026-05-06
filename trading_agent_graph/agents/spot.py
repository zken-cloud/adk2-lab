import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_spot_trade

spot_agent = Agent(
    name="spot_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    description="Executes spot market cryptocurrency trades (buying and selling assets).",
    instruction="""
    You are a spot trading execution assistant.
    Read the approved trade request details, validate the symbol and amount,
    and call the `execute_spot_trade` tool to process the market order.
    Summarize the final execution receipt for the user clearly.
    """,
    tools=[execute_spot_trade],
)
