from google.adk.workflow.node import node
from google.adk.events.event import Event
from google.genai import types

# ==============================================================================
# [ADK 2.0 FEATURE]: Pre-Processing @node Tool for Dynamic Workflows
# ==============================================================================
@node(name="compliance_check", rerun_on_resume=True)
def compliance_check_tool(node_input: types.Content) -> Event:
    """
    A pre-processing compliance check node implemented using the ADK 2.0 @node decorator.
    It intercepts incoming messages before handing them off to the dynamic task delegator.
    """
    text_input = ''.join(p.text for p in (node_input.parts or []) if p.text).lower()
    
    print(f"[Dynamic Compliance Check] Analyzing message: '{text_input}'")
    
    if "hack" in text_input or "illegal" in text_input:
        return Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text("🚨 [Compliance Error]: This request violates trading compliance standards and cannot be processed.")]
            )
        )
        
    return Event(output=text_input)


