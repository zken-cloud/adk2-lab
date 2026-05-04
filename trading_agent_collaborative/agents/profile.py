import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.agents import Agent
from shared.tools import get_user_profile
from shared.schemas import ProfileOutput

profile_agent = Agent(
    name='profile_agent',
    model="gemini-3.1-flash-lite-preview",
    mode="single_turn",
    output_schema=ProfileOutput,
    description="Researches and retrieves user profile details and account balances instantly without user interaction.",
    instruction="""
    You are a fast single-turn lookup agent.
    Invoke `get_user_profile` and instantly return the pure data back to the coordinator without user interaction.
    """,
    tools=[get_user_profile],
)
