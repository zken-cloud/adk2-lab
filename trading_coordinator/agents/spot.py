from google.adk.agents import Agent
from pydantic import BaseModel, Field

class SpotTradeInput(BaseModel):
    symbol: str = Field(description="The cryptocurrency pair symbol, e.g., 'BTC/USDT'")
    side: str = Field(description="Must be 'buy' or 'sell'")
    amount: float = Field(description="The amount of cryptocurrency to trade")

def execute_spot_trade(symbol: str, side: str, amount: float) -> str:
    """Executes a spot trade on the exchange market."""
    price = 95000.0 if "BTC" in symbol else 3500.0
    cost = amount * price
    return f"Executed {side} of {amount} {symbol}. Estimated cost: {cost} USDT."

spot_agent = Agent(
    name="spot_agent",
    model="gemini-3.1-flash-lite-preview",
    description="Executes spot market cryptocurrency trades (buying and selling assets).",
    instruction="""
    You are a spot trading execution assistant.
    Read the approved trade request details, validate the symbol and amount,
    and call the `execute_spot_trade` tool to process the market order.
    Summarize the final execution receipt for the user clearly.
    """,
    tools=[execute_spot_trade],
)
