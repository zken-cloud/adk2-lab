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

"""Shared sub-agent sample: same SingleLlmAgent in multiple tree positions.

Root agent routes to travel_agent or shopping_agent. Both share the
same search_agent (SingleLlmAgent) instance that wraps a search_web
tool. Tests that search_agent works correctly via different paths
and that events carry distinct node_paths
(root/travel/search vs root/shopping/search).
"""

from google.adk.workflow.agents.llm_agent import LlmAgent
from google.adk.workflow.agents.single_llm_agent import SingleLlmAgent
from google.genai import types

SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
]

GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.1,
    safety_settings=SAFETY_SETTINGS,
)


def search_web(query: str) -> str:
  """Search the web for information.

  Args:
    query: The search query.

  Returns:
    Search results as a string.
  """
  q = query.lower()
  if any(w in q for w in ('flight', 'hotel', 'trip', 'travel', 'vacation')):
    return (
        'Search results:\n'
        '1. NYC to London round-trip from $450 (Delta, non-stop)\n'
        '2. NYC to London round-trip from $520 (British Airways, non-stop)\n'
        '3. London hotels from $120/night (Holiday Inn, Westminster)'
    )
  if any(w in q for w in ('laptop', 'buy', 'price', 'product', 'shop')):
    return (
        'Search results:\n'
        '1. ThinkPad X1 Carbon - $949, 14" display, 16GB RAM\n'
        '2. MacBook Air M3 - $999, 13.6" display, 16GB RAM\n'
        '3. Dell XPS 13 - $899, 13.4" display, 16GB RAM'
    )
  return f'Search results for "{query}": No relevant results found.'


# Shared sub-agent: same instance reused in travel_agent and
# shopping_agent. Disallow transfers so it always calls the tool
# and reports results back to its parent.
search_agent = SingleLlmAgent(
    name='search_agent',
    description='Searches the web and returns raw results.',
    instruction="""\
You are a web search assistant. When given a query, call the
search_web tool and return the raw results. Do not interpret
or summarize — just return what search_web gives you.
""",
    tools=[search_web],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=GEN_CONFIG,
)

travel_agent = LlmAgent(
    name='travel_agent',
    description='Handles travel-related questions (flights, hotels, trips).',
    instruction="""\
You help with travel questions. Delegate to search_agent to find
flights, hotels, or travel information, then summarize the results.
""",
    sub_agents=[search_agent],
    generate_content_config=GEN_CONFIG,
)

shopping_agent = LlmAgent(
    name='shopping_agent',
    description='Handles shopping and product questions.',
    instruction="""\
You help with shopping questions. Delegate to search_agent to find
products and prices, then summarize the best options.
""",
    sub_agents=[search_agent],
    generate_content_config=GEN_CONFIG,
)

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='Routes questions to specialized agents.',
    instruction="""\
You route questions to the appropriate agent:
- Travel questions (flights, hotels, trips): delegate to travel_agent
- Shopping questions (products, prices): delegate to shopping_agent
Always delegate. Do not answer directly.
""",
    sub_agents=[travel_agent, shopping_agent],
    generate_content_config=GEN_CONFIG,
)
