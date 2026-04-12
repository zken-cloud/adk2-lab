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

"""Test script comparing SingleLlmAgent vs old LlmAgent behavior.

Runs both agents with the same tools and prompts to verify they produce
equivalent event sequences (function calls, function responses, text).
"""

import asyncio

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from .agent import check_prime
from .agent import roll_die
from .agent import root_agent as single_agent

load_dotenv(override=True)

USER_ID = "test_user"


def print_event(event, indent="  "):
  """Print event details in a readable format."""
  print(f"{indent}Event: author={event.author}, id={event.id}")
  if event.content and event.content.parts:
    for i, part in enumerate(event.content.parts):
      if part.text:
        print(f"{indent}  Part {i} [Text]: {part.text.strip()}")
      if part.function_call:
        print(
            f"{indent}  Part {i} [FunctionCall]:"
            f" {part.function_call.name}({part.function_call.args})"
        )
      if part.function_response:
        print(
            f"{indent}  Part {i} [FunctionResponse]:"
            f" {part.function_response.name}"
            f" -> {part.function_response.response}"
        )
  if event.actions and event.actions.transfer_to_agent:
    print(f"{indent}  Transfer to: {event.actions.transfer_to_agent}")


def summarize_events(events):
  """Summarize event sequence for comparison."""
  summary = []
  for event in events:
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.function_call:
        summary.append(f"FunctionCall:{part.function_call.name}")
      elif part.function_response:
        summary.append(f"FunctionResponse:{part.function_response.name}")
      elif part.text and part.text.strip():
        summary.append("Text")
  return summary


async def run_tests_for_agent(agent_label, runner, session_id):
  """Run standard test prompts and return collected events per test."""
  session_service = runner.session_service
  app_name = runner.app_name

  session = await session_service.create_session(
      app_name=app_name,
      user_id=USER_ID,
      session_id=session_id,
  )

  all_results = {}

  # Test 1: Simple tool call
  print(f"\n{'=' * 60}")
  print(f"Test 1 ({agent_label}): Roll a 6-sided die")
  print(f"{'=' * 60}")
  content = types.Content(
      role="user",
      parts=[types.Part(text="Roll a 6-sided die for me.")],
  )
  print(f'\n>>> User: "Roll a 6-sided die for me."')
  print("--- Running agent ---")
  events = []
  async for event in runner.run_async(
      session_id=session.id,
      user_id=USER_ID,
      new_message=content,
  ):
    print_event(event)
    events.append(event)
  print("--- End of turn ---\n")
  all_results["test1"] = events

  # Test 2: Multi-step tool call
  print(f"\n{'=' * 60}")
  print(f"Test 2 ({agent_label}): Roll and check prime")
  print(f"{'=' * 60}")
  content = types.Content(
      role="user",
      parts=[
          types.Part(
              text=(
                  "Roll an 8-sided die and check if the result is a"
                  " prime number."
              )
          )
      ],
  )
  print(
      f'\n>>> User: "Roll an 8-sided die and check if the result is a'
      f' prime number."'
  )
  print("--- Running agent ---")
  events = []
  async for event in runner.run_async(
      session_id=session.id,
      user_id=USER_ID,
      new_message=content,
  ):
    print_event(event)
    events.append(event)
  print("--- End of turn ---\n")
  all_results["test2"] = events

  return all_results


async def main():
  """Run both agents and compare event sequences."""

  # --- Old LlmAgent with Runner ---
  # Reuse the same tools/instruction/config as the SingleLlmAgent in agent.py
  llm_agent = Agent(
      model=single_agent.model,
      name=single_agent.name,
      description=single_agent.description,
      instruction=single_agent.instruction,
      tools=[roll_die, check_prime],
  )

  llm_session_service = InMemorySessionService()
  llm_runner = Runner(
      agent=llm_agent,
      app_name="workflow_hello_world_llm",
      session_service=llm_session_service,
  )

  print("\n" + "#" * 70)
  print("# Testing with old LlmAgent (Agent) + Runner")
  print("#" * 70)
  llm_results = await run_tests_for_agent("LlmAgent", llm_runner, "session_llm")

  # --- SingleLlmAgent with Runner ---
  workflow_app = App(
      name="workflow_hello_world_single",
      root_agent=single_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )

  workflow_session_service = InMemorySessionService()
  workflow_runner = Runner(
      app=workflow_app,
      session_service=workflow_session_service,
  )

  print("\n" + "#" * 70)
  print("# Testing with SingleLlmAgent + Runner")
  print("#" * 70)
  single_results = await run_tests_for_agent(
      "SingleLlmAgent", workflow_runner, "session_single"
  )

  # --- Compare event sequences ---
  print("\n" + "=" * 70)
  print("Comparison: Event Sequence Summary")
  print("=" * 70)

  for test_name in ["test1", "test2"]:
    llm_summary = summarize_events(llm_results[test_name])
    single_summary = summarize_events(single_results[test_name])
    match = llm_summary == single_summary
    print(f"\n  {test_name}:")
    print(f"    LlmAgent:       {llm_summary}")
    print(f"    SingleLlmAgent: {single_summary}")
    print(f"    Match: {'YES' if match else 'NO (see details above)'}")

  print("\n" + "=" * 70)
  print("All tests completed!")
  print("=" * 70)


if __name__ == "__main__":
  asyncio.run(main())
