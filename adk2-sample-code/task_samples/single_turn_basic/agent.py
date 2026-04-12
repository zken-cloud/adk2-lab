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

"""Single-turn basic: parent delegates to autonomous single-turn child.

Demonstrates the single-turn delegation pattern:
  1. User asks the coordinator to summarize a document.
  2. Coordinator calls request_task_summarizer(...) to delegate.
  3. Summarizer completes autonomously (no user interaction),
     calls finish_task with the result.
  4. Coordinator receives the result and responds to the user.

Single-turn agents never interact with the user directly. They
receive input, do their work, and return a result.

Run with:
  adk web contributing/task_samples/
"""

from google.adk.workflow.agents.llm_agent import Agent
from pydantic import BaseModel

# -- Schemas ----------------------------------------------------------------


class SummaryOutput(BaseModel):
  """Output schema for the summarizer agent."""

  summary: str
  word_count: int
  key_points: str


# -- Tools ------------------------------------------------------------------


def extract_text(url: str) -> str:
  """Extract text content from a URL.

  Args:
    url: The URL to extract text from.

  Returns:
    Simulated extracted text content.
  """
  return (
      f'Extracted content from {url}:\n'
      'Artificial intelligence (AI) has transformed industries'
      ' worldwide. Machine learning models now power search engines,'
      ' recommendation systems, and autonomous vehicles. Recent'
      ' advances in large language models have enabled new'
      ' applications in code generation, creative writing, and'
      ' scientific research. Experts predict continued rapid growth'
      ' in AI capabilities through 2026 and beyond.'
  )


def count_words(text: str) -> int:
  """Count the number of words in a text.

  Args:
    text: The text to count words in.

  Returns:
    The word count.
  """
  return len(text.split())


# -- Agents ----------------------------------------------------------------

summarizer = Agent(
    name='summarizer',
    mode='single_turn',
    output_schema=SummaryOutput,
    instruction=(
        'You are a document summarizer. When given a goal:\n'
        '1. Use extract_text to get the document content.\n'
        '2. Use count_words to count the words.\n'
        '3. Call finish_task with a concise summary, the word count,'
        ' and key points.\n'
        'You will NOT receive any user replies. Complete the task'
        ' using only the information provided.'
    ),
    description='Summarizes documents from URLs autonomously.',
    tools=[extract_text, count_words],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[summarizer],
    instruction=(
        'You are a helpful assistant. When the user asks you to'
        ' summarize a document or URL, delegate the work to the'
        ' summarizer agent using request_task_summarizer. After'
        ' the summarizer completes, present the summary, word'
        ' count, and key points to the user in a readable format.'
    ),
)
