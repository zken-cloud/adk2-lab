from google.adk.agents import Agent
from pydantic import BaseModel, Field

class TransferInput(BaseModel):
    asset: str = Field(description="Cryptocurrency asset symbol, e.g., 'USDT'")
    amount: float = Field(description="Amount to send")
    destination: str = Field(description="Blockchain address or wallet ID")

def execute_transfer(asset: str, amount: float, destination: str) -> str:
    """Transfers cryptocurrency assets to an external destination address."""
    return f"Transfer transaction confirmed: Sent {amount} {asset} to {destination}."

transfer_agent = Agent(
    name="transfer_agent",
    model="gemini-3.1-flash-lite-preview",
    description="Manages transferring, sending, and withdrawing cryptocurrency assets to external addresses.",
    instruction="""
    You are an asset transfer and withdrawal assistant.
    Read the approved transfer details, make sure a valid destination address is specified,
    and execute the `execute_transfer` tool.
    Report the transaction hash and success summary to the user.
    """,
    tools=[execute_transfer],
)
