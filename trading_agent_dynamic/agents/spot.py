import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_spot_trade
from shared.schemas import SpotTradeInput, SpotTradeOutput

spot_agent = Agent(
    name="spot_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    input_schema=SpotTradeInput,
    output_schema=SpotTradeOutput,
    description="Executes spot market cryptocurrency trades (buying and selling standard assets).",
    instruction="""
    You are a specialized spot trading execution agent.
    Read the parameters from your input schema, invoke the `execute_spot_trade` tool,
    and call finish_task to report the receipt back to the master coordinator.
    """,
    tools=[execute_spot_trade],
)
