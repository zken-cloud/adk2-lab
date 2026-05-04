import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_transfer

transfer_agent = Agent(
    name="transfer_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="chat",
    description="Manages transferring, sending, and withdrawing cryptocurrency assets to external blockchain addresses.",
    instruction="""
    You are a collaborative external asset transfer expert.
    Help the user send crypto off-platform. Validate that they have provided a valid destination address. If missing, ask for it.
    Execute `execute_transfer` and clearly provide the summary hash back to the user.
    """,
    tools=[execute_transfer],
)
