from google.adk.agents import Agent
from pydantic import BaseModel

class ProfileOutput(BaseModel):
    name: str
    kyc_status: str
    membership_tier: str
    balances: dict

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
    mode="task",
    output_schema=ProfileOutput,
    description="Researches and retrieves user profile details, KYC verification status, and wallet balances.",
    instruction="""
    You are a specialized financial account data retrieval agent.
    Invoke the `get_user_profile` tool to look up the user's current balances and KYC status.
    Then securely call finish_task to return the data to the master coordinator.
    """,
    tools=[get_user_profile],
)
