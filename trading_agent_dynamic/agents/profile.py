import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import get_user_profile
from shared.schemas import ProfileInput, ProfileOutput

profile_agent = Agent(
    name='profile_agent',
    model="gemini-3.1-flash-lite-preview",
    mode="task",
    input_schema=ProfileInput,
    output_schema=ProfileOutput,
    description="Researches and retrieves user profile details, KYC verification status, and wallet balances.",
    instruction="""
    You are a specialized financial account data retrieval agent.
    Invoke the `get_user_profile` tool to look up the user's current balances and KYC status.
    Then securely call finish_task to return the data to the master coordinator.
    """,
    tools=[get_user_profile],
)
