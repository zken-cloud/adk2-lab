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
    description="Handles user profile checking, KYC verification status, and account balance inquiries.",
    instruction="""
    You are a customer account and profile agent.
    Call the `get_user_profile` tool to inspect the current user's KYC status and wallet balances.
    Provide a highly organized and professional financial summary back to the user.
    """,
    tools=[get_user_profile],
)
