import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from google.adk.workflow import node
from google.adk.events.event import Event
from google.genai import types
from shared.tools import check_kyc_status

@node(name="compliance_check", rerun_on_resume=True)
def compliance_check_tool(node_input: types.Content) -> Event:
    """Pre-processing KYC compliance check. Warns if user is not verified but allows the request through."""
    kyc = check_kyc_status()

    print(f"[Compliance Check] User: {kyc['user_name']}, KYC Status: {kyc['kyc_status']}")

    if not kyc["verified"]:
        return Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=kyc["message"])]
            ),
            output=node_input,
            route="pass",
        )

    return Event(output=node_input, route="pass")
