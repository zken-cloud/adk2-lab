from google.adk.agents import Agent

def execute_transfer(asset: str, amount: float, destination: str) -> str:
    """Transfers cryptocurrency assets to an external destination address."""
    return f"Transfer transaction confirmed: Sent {amount} {asset} to {destination}."

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
