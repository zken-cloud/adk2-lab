from google.adk.agents import Agent

def execute_derivative_trade(symbol: str, side: str, amount: float, leverage: int) -> str:
    """Executes a leveraged futures position on the derivatives platform."""
    return f"Successfully opened a {side} position on {symbol} for {amount} units at {leverage}x leverage."

derivative_agent = Agent(
    name="derivative_agent",
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    description="Handles perpetual futures, margin leverage trading, and opening derivative positions via structured execution.",
    instruction="""
    You are a structured task execution agent.
    Execute the requested leveraged derivative order and securely hand the execution receipt back to the parent.
    """,
    tools=[execute_derivative_trade],
)
