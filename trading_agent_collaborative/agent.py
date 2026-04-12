import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig

from agents.profile import profile_agent
from agents.spot import spot_agent
from agents.derivative import derivative_agent
from agents.transfer import transfer_agent

# ==============================================================================
# [ADK 2.0 FEATURE]: Mixed-Mode Orchestration
# We combine multiple delegation strategies under a single conversational root:
# 1. `profile_agent`: mode='single_turn' (instant autonomous lookup)
# 2. `derivative_agent`: mode='task' (structured guardrails)
# 3. `spot_agent` / `transfer_agent`: mode='chat' (conversational hand-off)
# ==============================================================================
root_agent = Agent(
    name="trading_coordinator_collaborative",
    model="gemini-3.1-flash-lite-preview",
    mode="chat",
    description="Master collaborative gateway that directs incoming user chats to the most suitable specialized team member.",
    instruction="""
    You are the highly conversational collaborative multi-agent crypto coordinator.
    Your job is to welcome the user, assess their intent, and gracefully transfer the chat conversation directly to the best available expert in your team:
    
    - If they want profile or account details, transfer to profile_agent.
    - If they want to buy/sell crypto instantly, transfer to spot_agent.
    - If they want perpetual leverage or margin, transfer to derivative_agent.
    - If they want to withdraw or send funds off-platform, transfer to transfer_agent.
    
    Ensure you provide a warm greeting and hand them off properly.
    """,
    sub_agents=[
        profile_agent,
        spot_agent,
        derivative_agent,
        transfer_agent,
    ],
)

app = App(
    name="trading_agent_collaborative",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

agent = root_agent
