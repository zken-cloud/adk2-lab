import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import get_user_profile

profile_agent = Agent(
    name='profile_agent',
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    description="Handles user profile checking, KYC verification status, and account balance inquiries.",
    instruction="""
    You are a customer account and profile agent.
    Call the `get_user_profile` tool to inspect the current user's KYC status and wallet balances.
    Provide a highly organized and professional financial summary back to the user.
    """,
    tools=[get_user_profile],
)
