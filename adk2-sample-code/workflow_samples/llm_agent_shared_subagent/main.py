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

"""Test script verifying shared SingleLlmAgent across an agent tree.

The same search_agent (SingleLlmAgent) instance is used as a sub-agent
of both travel_agent and shopping_agent. This test verifies:
1. search_agent works correctly when reached via different paths
   (root -> travel -> search vs root -> shopping -> search).
2. Events from search_agent carry distinct node_paths per path.
"""

import asyncio

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from .agent import root_agent

load_dotenv(override=True)

USER_ID = 'user_1'


def print_event(event, indent='  '):
  """Print event details in a readable format."""
  node_path = getattr(event, 'node_path', None)
  path_str = f', node_path={node_path}' if node_path else ''
  print(f'{indent}Event: author={event.author}{path_str}')
  if event.content and event.content.parts:
    for i, part in enumerate(event.content.parts):
      if part.text:
        print(f'{indent}  Part {i} [Text]: {part.text.strip()}')
      if part.function_call:
        print(
            f'{indent}  Part {i} [FunctionCall]:'
            f' {part.function_call.name}'
            f'({part.function_call.args})'
        )
      if part.function_response:
        print(
            f'{indent}  Part {i} [FunctionResponse]:'
            f' {part.function_response.name}'
            f' -> {part.function_response.response}'
        )
  if event.actions and event.actions.transfer_to_agent:
    print(f'{indent}  Transfer to: {event.actions.transfer_to_agent}')


def summarize_events(events):
  """Summarize event sequence for comparison."""
  summary = []
  for event in events:
    parts_summary = []
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call:
          parts_summary.append(f'FunctionCall:{part.function_call.name}')
        elif part.function_response:
          parts_summary.append(
              f'FunctionResponse:{part.function_response.name}'
          )
        elif part.text and part.text.strip():
          parts_summary.append('Text')
    if event.actions and event.actions.transfer_to_agent:
      parts_summary.append(f'Transfer:{event.actions.transfer_to_agent}')
    if parts_summary:
      summary.append(f'{event.author}:[{",".join(parts_summary)}]')
  return summary


def extract_node_paths(events, author):
  """Extract unique node_paths for events from a specific author."""
  paths = set()
  for event in events:
    node_path = getattr(event, 'node_path', None)
    if event.author == author and node_path:
      paths.add(node_path)
  return paths


async def run_tests(runner, session_service, app_name):
  """Run shared sub-agent tests.

  Returns:
    Dict mapping test name to list of events.
  """
  all_results = {}

  # Test 1: Travel path (root -> travel_agent -> search_agent)
  session1 = await session_service.create_session(
      app_name=app_name,
      user_id=USER_ID,
      session_id='session_travel',
  )

  print(f"\n{'=' * 60}")
  print('Test 1: Travel path (root -> travel -> search)')
  print(f"{'=' * 60}")
  content = types.Content(
      role='user',
      parts=[types.Part(text='Find me a flight from NYC to London.')],
  )
  print('\n>>> User: "Find me a flight from NYC to London."')
  print('--- Running agent ---')
  events = []
  async for event in runner.run_async(
      session_id=session1.id,
      user_id=USER_ID,
      new_message=content,
  ):
    print_event(event)
    events.append(event)
  print('--- End of turn ---\n')
  all_results['test1'] = events

  # Test 2: Shopping path (root -> shopping_agent -> search_agent)
  session2 = await session_service.create_session(
      app_name=app_name,
      user_id=USER_ID,
      session_id='session_shopping',
  )

  print(f"\n{'=' * 60}")
  print('Test 2: Shopping path (root -> shopping -> search)')
  print(f"{'=' * 60}")
  content = types.Content(
      role='user',
      parts=[types.Part(text='Find a good laptop under $1000.')],
  )
  print('\n>>> User: "Find a good laptop under $1000."')
  print('--- Running agent ---')
  events = []
  async for event in runner.run_async(
      session_id=session2.id,
      user_id=USER_ID,
      new_message=content,
  ):
    print_event(event)
    events.append(event)
  print('--- End of turn ---\n')
  all_results['test2'] = events

  return all_results


async def main():
  """Run tests and verify node_paths."""

  session_service = InMemorySessionService()
  runner = Runner(
      agent=root_agent,
      app_name='shared_subagent',
      session_service=session_service,
  )

  print('\n' + '#' * 70)
  print('# Testing shared SingleLlmAgent with workflow LlmAgent')
  print('#' * 70)
  results = await run_tests(runner, session_service, 'shared_subagent')

  # --- Event summaries ---
  print('\n' + '=' * 70)
  print('Event Sequence Summary')
  print('=' * 70)

  for test_name in ['test1', 'test2']:
    summary = summarize_events(results[test_name])
    print(f'\n  {test_name}: {summary}')

  # --- Verify node_paths ---
  print('\n' + '=' * 70)
  print('Node Path Verification')
  print('=' * 70)

  test1_paths = extract_node_paths(results['test1'], 'search_agent')
  test2_paths = extract_node_paths(results['test2'], 'search_agent')
  print(f'\n  Test 1 search_agent node_paths: {test1_paths}')
  print(f'  Test 2 search_agent node_paths: {test2_paths}')

  if test1_paths and test2_paths:
    paths_differ = test1_paths != test2_paths
    print('  Paths differ:' f" {'YES' if paths_differ else 'NO (UNEXPECTED)'}")
    test1_has_travel = any('travel_agent' in p for p in test1_paths)
    test2_has_shopping = any('shopping_agent' in p for p in test2_paths)
    print(f'  Test 1 contains "travel_agent": {test1_has_travel}')
    print(f'  Test 2 contains "shopping_agent": {test2_has_shopping}')
  else:
    print('  WARNING: No node_paths found for search_agent')

  print('\n' + '=' * 70)
  print('All tests completed!')
  print('=' * 70)


if __name__ == '__main__':
  asyncio.run(main())
