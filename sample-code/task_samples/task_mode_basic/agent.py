# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task mode basic: parent delegates research to a task-mode child.

Demonstrates the full task delegation cycle:
  1. User asks the coordinator to research a topic.
  2. Coordinator calls request_task_researcher(...) to delegate.
  3. Researcher works on the task (can chat with the user for
     clarification) and calls finish_task when done.
  4. Coordinator receives the result and responds to the user.

Run with:
  adk web contributing/task_samples/
"""

from google.adk.workflow.agents.llm_agent import Agent
from pydantic import BaseModel

# -- Schemas ----------------------------------------------------------------


class ResearchInput(BaseModel):
  """Input schema for the researcher agent."""

  topic: str
  depth: str = 'standard'


class ResearchOutput(BaseModel):
  """Output schema for the researcher agent."""

  summary: str
  key_findings: str
  confidence: str


# -- Tools ------------------------------------------------------------------


def search_web(query: str) -> str:
  """Search the web for information on a query.

  Args:
    query: The search query string.

  Returns:
    Simulated search results.
  """
  return (
      f'Search results for "{query}":\n'
      f'1. Overview of {query} from Wikipedia\n'
      f'2. Recent developments in {query} (2026)\n'
      f'3. Expert analysis: the future of {query}'
  )


def analyze_sources(sources: str) -> str:
  """Analyze and synthesize information from multiple sources.

  Args:
    sources: The source material to analyze.

  Returns:
    An analysis summary.
  """
  return (
      f'Analysis complete. Synthesized {len(sources.split())} words '
      'of source material into key findings.'
  )


# -- Agents ----------------------------------------------------------------

researcher = Agent(
    name='researcher',
    mode='task',
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    instruction=(
        'You are a thorough research assistant. When given a topic:\n'
        '1. Use search_web to find relevant information.\n'
        '2. Use analyze_sources to synthesize your findings.\n'
        '3. If the user asks for clarification or changes, adjust your'
        ' research accordingly.\n'
        '4. When you have completed the research, call finish_task with'
        ' a summary, key_findings, and confidence level.'
    ),
    description='Researches topics using web search and analysis tools.',
    tools=[search_web, analyze_sources],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[researcher],
    instruction=(
        'You are a helpful coordinator. When the user asks you to'
        ' research something, delegate the work to the researcher'
        ' agent using request_task_researcher. After the researcher'
        ' completes the task, summarize the results for the user in'
        ' a clear and concise way.'
    ),
)
