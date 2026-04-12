from google.adk.agents import Agent
from pydantic import BaseModel, Field

class SpotTradeInput(BaseModel):
    symbol: str = Field(description="The cryptocurrency pair symbol, e.g., 'BTC/USDT'")
    side: str = Field(description="Must be 'buy' or 'sell'")
    amount: float = Field(description="The amount of cryptocurrency to trade")

class SpotTradeOutput(BaseModel):
    execution_receipt: str
    estimated_cost: str

def execute_spot_trade(symbol: str, side: str, amount: float) -> str:
    """Executes a spot trade on the exchange market."""
    price = 95000.0 if "BTC" in symbol else 3500.0
    cost = amount * price
    return f"Executed {side} of {amount} {symbol}. Estimated cost: {cost} USDT."

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
