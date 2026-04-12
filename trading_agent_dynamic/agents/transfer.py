from google.adk.agents import Agent
from pydantic import BaseModel, Field

class TransferInput(BaseModel):
    asset: str = Field(description="Cryptocurrency asset symbol, e.g., 'USDT'")
    amount: float = Field(description="Amount to send")
    destination: str = Field(description="Blockchain address or wallet ID")

class TransferOutput(BaseModel):
    transaction_receipt: str

def execute_transfer(asset: str, amount: float, destination: str) -> str:
    """Transfers cryptocurrency assets to an external destination address."""
    return f"Transfer transaction confirmed: Sent {amount} {asset} to {destination}."

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
