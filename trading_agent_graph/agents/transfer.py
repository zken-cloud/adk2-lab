import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_transfer

transfer_agent = Agent(
    name="transfer_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    description="Manages transferring, sending, and withdrawing cryptocurrency assets to external addresses.",
    instruction="""
    You are an asset transfer and withdrawal assistant.
    Read the approved transfer details, make sure a valid destination address is specified,
    and execute the `execute_transfer` tool.
    Report the transaction hash and success summary to the user.
    """,
    tools=[execute_transfer],
)
