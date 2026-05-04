import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_spot_trade

spot_agent = Agent(
    name="spot_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="chat",
    description="Executes spot market cryptocurrency trades (buying and selling standard assets like BTC, ETH, USDT).",
    instruction="""
    You are a collaborative spot trading execution expert.
    Read the user's desired order parameters. If any parameters (symbol, side, amount) are missing, kindly chat with the user to clarify them.
    Once everything is clear, call `execute_spot_trade` and summarize the execution receipt.
    """,
    tools=[execute_spot_trade],
)
