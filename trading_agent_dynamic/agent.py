import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json, re
from google import genai
from google.adk import Context, Workflow
from google.adk.workflow import node
from google.adk.events.event import Event
from google.adk.apps.app import App, ResumabilityConfig
from google.genai import types

from shared.tools import (
    get_user_profile,
    execute_spot_trade,
    execute_derivative_trade,
    execute_transfer,
    check_kyc_status,
)

# --- Function nodes ---

@node
def profile_node(node_input):
    p = get_user_profile()
    balances = ", ".join(f"{v} {k}" for k, v in p.get("balances", {}).items())
    return f"Name: {p['name']}, KYC: {p['kyc_status']}, Tier: {p['tier']}, Balances: {balances}"

@node
def spot_node(node_input: dict):
    return execute_spot_trade(node_input.get("symbol", "BTC/USDT"), node_input.get("side", "buy"), node_input.get("amount", 1.0))

@node
def derivative_node(node_input: dict):
    return execute_derivative_trade(node_input.get("symbol", "ETH-PERP"), node_input.get("side", "long"), node_input.get("amount", 1.0), node_input.get("leverage", 5))

@node
def transfer_node(node_input: dict):
    return execute_transfer(node_input.get("asset", "USDT"), node_input.get("amount", 100.0), node_input.get("destination", "unknown"))

INTENT_PROMPT = """You are an intent parser. Return ONLY a JSON object:
{"intent": "profile|spot|derivative|transfer|unknown", "params": {...}}
Examples:
- "Check my account" -> {"intent": "profile", "params": {}}
- "Buy 2 BTC" -> {"intent": "spot", "params": {"symbol": "BTC/USDT", "side": "buy", "amount": 2.0}}
- "Open 10x long ETH-PERP" -> {"intent": "derivative", "params": {"symbol": "ETH-PERP", "side": "long", "amount": 1.0, "leverage": 10}}
- "Send 500 USDT to 0xabc123" -> {"intent": "transfer", "params": {"asset": "USDT", "amount": 500.0, "destination": "0xabc123"}}"""

@node
async def parse_intent(node_input: str) -> dict:
    resp = await genai.Client().aio.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"{INTENT_PROMPT}\n\nUser: {node_input}",
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
    )
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", resp.text.strip())
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"intent": "unknown", "params": {}}

# --- Dynamic workflow ---

ROUTE_MAP = {
    "profile": profile_node,
    "spot": spot_node,
    "derivative": derivative_node,
    "transfer": transfer_node,
}

@node(rerun_on_resume=True)
async def dynamic_trading_workflow(ctx: Context, node_input: str):
    kyc = check_kyc_status()
    print(f"[Compliance Check] User: {kyc['user_name']}, KYC Status: {kyc['kyc_status']}")
    kyc_warning = f"\n\n{kyc['message']}" if not kyc["verified"] else ""

    parsed = await ctx.run_node(parse_intent, node_input)
    target = ROUTE_MAP.get(parsed.get("intent"))

    if target:
        result = await ctx.run_node(target, parsed.get("params", {}))
        text = f"{result}{kyc_warning}"
    else:
        text = f"I can help with: profile lookups, spot trading, derivative trading, and transfers.{kyc_warning}"

    return Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        output=text,
    )

root_agent = Workflow(
    name="trading_agent_dynamic",
    edges=[("START", dynamic_trading_workflow)],
)

app = App(
    name="trading_agent_dynamic",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

agent = root_agent
