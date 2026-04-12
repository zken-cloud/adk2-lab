from google.adk.agents import Agent
from pydantic import BaseModel, Field

class DerivativeTradeInput(BaseModel):
    symbol: str = Field(description="The futures symbol, e.g., 'ETH-PERP'")
    side: str = Field(description="Must be 'long' or 'short'")
    amount: float = Field(description="Position size")
    leverage: int = Field(description="Leverage multiplier")

class DerivativeTradeOutput(BaseModel):
    confirmation: str

def execute_derivative_trade(symbol: str, side: str, amount: float, leverage: int) -> str:
    """Executes a leveraged futures position on the derivatives platform."""
    return f"Successfully opened a {side} position on {symbol} for {amount} units at {leverage}x leverage."

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    input_schema=DerivativeTradeInput,
    output_schema=DerivativeTradeOutput,
    description="Handles perpetual futures, leverage trading, and opening derivative positions.",
    instruction="""
    You are a specialized derivatives execution agent.
    Read the parameters from your input schema, invoke `execute_derivative_trade`,
    and securely invoke finish_task to return the receipt to the coordinator.
    """,
    tools=[execute_derivative_trade],
)
