import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import execute_transfer
from shared.schemas import TransferInput, TransferOutput

transfer_agent = Agent(
    name="transfer_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    input_schema=TransferInput,
    output_schema=TransferOutput,
    description="Manages transferring, sending, and withdrawing cryptocurrency assets to external addresses.",
    instruction="""
    You are a specialized asset transfer and withdrawal agent.
    Read the parameters from your input schema, invoke `execute_transfer`,
    and invoke finish_task to send the receipt to the coordinator.
    """,
    tools=[execute_transfer],
)
