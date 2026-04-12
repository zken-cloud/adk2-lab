from google.adk.agents import Agent

def execute_spot_trade(symbol: str, side: str, amount: float) -> str:
    """Executes a spot trade on the exchange market."""
    price = 95000.0 if "BTC" in symbol else 3500.0
    cost = amount * price
    return f"Executed {side} of {amount} {symbol}. Estimated cost: {cost} USDT."

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
