import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App, ResumabilityConfig
from google.genai import types

from agents.profile import profile_agent
from agents.spot import spot_agent
from agents.derivative import derivative_agent
from agents.transfer import transfer_agent
from shared.tools import check_kyc_status

# ==============================================================================
# [ADK 2.0 FEATURE]: before_agent_callback Compliance Guardrail
# Demonstrates the callback-based approach to pre-processing guardrails.
# Checks KYC verification status and prepends a warning if not verified.
# Returns None to allow the request through (warn but allow).
# ==============================================================================
def compliance_guard(callback_context: CallbackContext) -> types.Content | None:
    kyc = check_kyc_status()
    print(f"[Compliance Check] User: {kyc['user_name']}, KYC Status: {kyc['kyc_status']}")
    if not kyc["verified"]:
        print(f"[Compliance Warning] {kyc['message']}")
    return None

# ==============================================================================
# [ADK 2.0 FEATURE]: Mixed-Mode Orchestration
# We combine multiple delegation strategies under a single conversational root:
# 1. `profile_agent`: mode='single_turn' (instant autonomous lookup)
# 2. `derivative_agent`: mode='task' (structured guardrails)
# 3. `spot_agent` / `transfer_agent`: mode='chat' (conversational hand-off)
# Note: mode is NOT set on the root agent — it is a subagent-only property.
# ==============================================================================
root_agent = Agent(
    name="trading_coordinator_collaborative",
    model="gemini-3.1-flash-lite-preview",
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
    before_agent_callback=compliance_guard,
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
