from google.adk.agents import Agent
from pydantic import BaseModel, Field

class DerivativeTradeInput(BaseModel):
    symbol: str = Field(description="The futures symbol, e.g., 'ETH-PERP'")
    side: str = Field(description="Must be 'long' or 'short'")
    amount: float = Field(description="Position size")
    leverage: int = Field(description="Leverage multiplier")

def execute_derivative_trade(symbol: str, side: str, amount: float, leverage: int) -> str:
    """Executes a leveraged futures position on the derivatives platform."""
    return f"Successfully opened a {side} position on {symbol} for {amount} units at {leverage}x leverage."

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    description="Handles perpetual futures, leverage trading, and opening derivative positions.",
    instruction="""
    You are a derivatives and perpetual futures trading assistant.
    Review the approved derivatives order, confirm the leverage factor is safe (1x to 20x max),
    and invoke the `execute_derivative_trade` tool to open the position.
    Output the final position confirmation to the user.
    """,
    tools=[execute_derivative_trade],
)
