"""Sample workflow for simple sequential workflow with LLM agents."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import Edge
from google.adk.workflow import Workflow

generate_fruit_agent = LlmAgent(
    name="generate_fruit_agent",
    model="gemini-2.5-flash",
    instruction="""Return the name of a random fruit.
      Return only the name, nothing else.""",
)

generate_benefit_agent = LlmAgent(
    name="generate_benefit_agent",
    model="gemini-2.5-flash",
    instruction="""Tell me a health benefit about the specified fruit.""",
)


root_agent = Workflow(
    name="root_agent",
    edges=Edge.chain("START", generate_fruit_agent, generate_benefit_agent),
)
