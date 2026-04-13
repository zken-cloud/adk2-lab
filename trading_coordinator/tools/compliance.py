from pydantic import BaseModel
from google.adk.workflow.node import node
from google.adk.events.event import Event
from google.genai import types

class ComplianceInput(BaseModel):
    user_input: str

# ==============================================================================
# [ADK 2.0 FEATURE]: Pre-Processing @node Tool
# ==============================================================================
@node(name="compliance_check", rerun_on_resume=True)
def compliance_check_tool(node_input: types.Content) -> Event:
    """
    A pre-processing compliance check node implemented using the ADK 2.0 @node decorator.
    It reads the initial user message and ensures it passes compliance before intent routing.
    """
    # Extract raw text from standard ADK 2.0 content types
    text_input = ''.join(p.text for p in (node_input.parts or []) if p.text).lower()
    
    print(f"[Compliance Check] Analyzing message: '{text_input}'")
    
    # Simple simulated flag detection
    if "hack" in text_input or "illegal" in text_input:
        # Flagged: stop execution by returning a default message
        return Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text("🚨 [Compliance Error]: This request violates trading compliance standards and cannot be processed.")]
            ),
            # This terminates the flow gracefully without downstream triggers
        )
        
    # Safe: continue by forwarding the parsed JSON-serializable text down the graph
    return Event(output=text_input)


