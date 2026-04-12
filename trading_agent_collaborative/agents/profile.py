from google.adk.agents import Agent

MOCK_USERS = {
    "user_123": {
        "name": "Alice Smith",
        "kyc_status": "Verified",
        "tier": "Premium",
        "balances": {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0}
    }
}

def get_user_profile(user_id: str = "user_123") -> dict:
    """Retrieves account balances, KYC status, and membership tier."""
    return MOCK_USERS.get(user_id, {"error": "User not found"})

profile_agent = Agent(
    name='profile_agent',
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    description="Researches and retrieves user profile details and account balances instantly without user interaction.",
    instruction="""
    You are a fast single-turn lookup agent.
    Invoke `get_user_profile` and instantly return the pure data back to the coordinator without user interaction.
    """,
    tools=[get_user_profile],
)
