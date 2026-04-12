from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.workflow.workflow_agent import WorkflowAgent
from google.adk.agents.workflow.base_node import START
from google.adk.agents.workflow.events.event import Event
# This import doesn't exist - from google.adk.models.content import Content
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types

process_message = LlmAgent(
   name="process_message",
   model="gemini-2.5-flash",
   instruction="""Classify user message into either "BUG", "CUSTOMER_SUPPORT", or "LOGISTICS".
   If you think a message applies to more than one category, reply with a comma separated list of categories.
   """,
)

def router(node_input: types.Content):
    try:
        # Attempt to parse the routes from the node input.
        text = node_input.parts[0].text # type: ignore
        if not text:
            # Raise an exception if the text is None or empty.
            raise ValueError("No text content in the input part.")

        routes = [route.strip() for route in text.split(",")]
        # Filter out any empty strings that may result from splitting.
        routes = [r for r in routes if r]

        if not routes:
            # Raise an exception if no valid routes are found after parsing.
            raise ValueError("No valid routes found after parsing.")

        # route can hold a list of RouteValue items
        yield Event(route=routes)

    except (AttributeError, IndexError, ValueError):
        # If any step in the try block fails, default to customer support.
        yield """
            Could not determine route from the input.
            Defaulting to customer support.
        """
        yield Event(route="CUSTOMER_SUPPORT")

def response_1_bug():
    # Can also just yield the string literal e.g.
    # yield "Handling bug..."
    yield types.ModelContent("Handling bug...")

def response_2_customer_support():
    # Can also just yield the string literal e.g.
    # yield "Handling customer support..."
    yield types.ModelContent("Handling customer support...")

def response_3_logistics():
    # Can also just yield the string literal e.g.
    # yield "Handling logistics..."
    yield types.ModelContent("Handling logistics...")

root_agent = WorkflowAgent(
   name="routing_workflow",
   edges=[
       # Sequential chain of nodes
       ("START", process_message, router),
       
       # Single edges from the router, route labelled in thrid position
       (router, response_1_bug, "BUG"),
       (router, response_2_customer_support, "CUSTOMER_SUPPORT"),
       (router, response_3_logistics, "LOGISTICS"),
   ],
)
