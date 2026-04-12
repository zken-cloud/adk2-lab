import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from google.adk.agents import Agent
from google.adk.apps.app import App, ResumabilityConfig

from agents.profile import profile_agent
from agents.spot import spot_agent
from agents.derivative import derivative_agent
from agents.transfer import transfer_agent

root_agent = Agent(
    name="trading_coordinator",
    model="gemini-3.1-flash-lite-preview",
    description="Master dynamic coordinator that automatically delegates requests to specialized child task agents.",
    instruction="""
    You are the dynamic multi-agent crypto trading coordinator.
    Your job is to read the user's request, identify the necessary actions, and call the appropriate auto-generated request_task tools to solve the user's request:

    - Profile queries: call `request_task_profile_agent`
    - Spot purchases/sales: call `request_task_spot_agent`
    - Futures/Leverage: call `request_task_derivative_agent`
    - Asset transfers: call `request_task_transfer_agent`

    Wait for the sub-agent to complete its task and report back before summarizing the final result to the user.
    """,
    sub_agents=[
        profile_agent,
        spot_agent,
        derivative_agent,
        transfer_agent,
    ],
)

# ==============================================================================
# [ADK 2.0 FEATURE]: Declarative App Container
# ==============================================================================
app = App(
    name="trading_coordinator_dynamic",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

agent = root_agent
