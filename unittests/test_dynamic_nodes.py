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

"""Tests for dynamic node execution."""

import asyncio
import time

from google.adk.agents.context import Context
from google.adk.apps import app
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.utils.workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
import pytest

from . import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_event_with_node
from .workflow_testing_utils import simplify_events_with_node


@pytest.mark.asyncio
async def test_dynamically_run_nodes_return_outputs_and_emit_events(
    request: pytest.FixtureRequest,
):
  """Tests a simple dynamic node execution where C calls A and B."""

  def func_a() -> str:
    return 'A'

  def func_b() -> str:
    return 'B'

  node_a = FunctionNode(func=func_a)
  node_b = FunctionNode(func=func_b)

  async def node_c(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C calls {output_a} & {output_b}'

  node_c = FunctionNode(func=node_c, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_dynamic_simple',
      edges=[
          (START, node_c),
      ],
  )
  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'func_a', 'output': 'A'},
      ),
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'func_b', 'output': 'B'},
      ),
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'node_c', 'output': 'C calls A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_with_custom_name(
    request: pytest.FixtureRequest,
):
  """Tests that custom node names can be provided for dynamic nodes."""

  def func_a() -> str:
    return 'A'

  node_a = FunctionNode(func=func_a)

  async def func_b(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a, name='custom_node_a')
    return f'B calls {output_a}'

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_custom_name',
      edges=[
          (START, node_b),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'test_agent_dynamic_custom_name',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'test_agent_dynamic_custom_name',
          {'node_name': 'custom_node_a', 'output': 'A'},
      ),
      (
          'test_agent_dynamic_custom_name',
          {'node_name': 'func_b', 'output': 'B calls A'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_hitl_no_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """Tests a dynamic node that requests input and doesn't rerun on resume.

  Given:
    A workflow where 'simple_caller' dynamically calls 'node_hitl', which
    requests human input via RequestInput. 'node_hitl' has
    rerun_on_resume=False.
  When:
    The workflow is executed, encounters RequestInput, pauses, and is then
    resumed with a user-provided response.
  Then:
    The workflow should pause at 'node_hitl', and upon resuming, it
    should complete successfully, with 'simple_caller' returning the
    user-provided input, without 'node_hitl' being rerun.
  """

  async def node_hitl():
    yield RequestInput(
        interrupt_id='req1',
        message='request 1',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def simple_caller(ctx: Context):
    result = await ctx.run_node(node_hitl)
    yield result
    yield 'parent done'

  simple_caller = FunctionNode(func=simple_caller, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_hitl',
      edges=[(START, simple_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req1',
              name='user_input',
              response={'text': 'Hello'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Check result
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_dynamic_hitl',
          {'node_name': 'node_hitl', 'output': {'text': 'Hello'}},
      ),
      (
          'test_agent_dynamic_hitl',
          {'node_name': 'simple_caller', 'output': {'text': 'Hello'}},
      ),
      (
          'test_agent_dynamic_hitl',
          {'node_name': 'simple_caller', 'output': 'parent done'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_hitl_with_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """Tests a dynamic node that requests input and reruns on resume.

  Given:
    A workflow where 'simple_caller' dynamically calls 'node_hitl', which
    requests human input via RequestInput. 'node_hitl' has
    rerun_on_resume=True and logic to handle resume inputs.
  When:
    The workflow is executed, encounters RequestInput, pauses, and is then
    resumed with a user-provided response.
  Then:
    The workflow should pause at 'node_hitl', and upon resuming, 'node_hitl'
    should be rerun, process the resume_input, and yield the contained
    value. 'simple_caller' should return this value.
  """

  async def node_hitl(ctx: Context):
    if resume_input := ctx.resume_inputs.get('req1'):
      yield resume_input['text']
      return

    yield RequestInput(
        interrupt_id='req1',
        message='request 1',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl, rerun_on_resume=True)

  async def simple_caller(ctx: Context) -> types.Content:
    result = await ctx.run_node(node_hitl)
    yield result
    yield 'parent done'

  simple_caller = FunctionNode(func=simple_caller, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_hitl_rerun',
      edges=[(START, simple_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req1',
              name='user_input',
              response={'text': 'Hello Rerun'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Check result
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_dynamic_hitl_rerun',
          {'node_name': 'node_hitl', 'output': 'Hello Rerun'},
      ),
      (
          'test_agent_dynamic_hitl_rerun',
          {'node_name': 'simple_caller', 'output': 'Hello Rerun'},
      ),
      (
          'test_agent_dynamic_hitl_rerun',
          {'node_name': 'simple_caller', 'output': 'parent done'},
      ),
  ]


@pytest.mark.asyncio
async def test_nested_dynamic_node_hitl(request: pytest.FixtureRequest):
  """Tests nested dynamic nodes with HITL.

  Given:
    A workflow with 'top_node' dynamically calling 'middle_node', which
    dynamically calls 'leaf_node_hitl'. 'leaf_node_hitl' requests human
    input via RequestInput.
  When:
    The workflow is executed, encounters RequestInput in 'leaf_node_hitl',
    pauses, and is then resumed with a user-provided response.
  Then:
    The workflow should pause at 'leaf_node_hitl', and upon resuming, all
    nodes ('leaf_node_hitl', 'middle_node', 'top_node') should complete
    successfully, propagating the user-provided input as their output.
  """

  async def leaf_node_hitl() -> str:
    yield RequestInput(
        interrupt_id='req2',
        message='request 2',
        response_schema={'type': 'string'},
    )

  async def middle_node(ctx: Context) -> str:
    return await ctx.run_node(leaf_node_hitl)

  middle_node = FunctionNode(func=middle_node, rerun_on_resume=True)
  leaf_node_hitl = FunctionNode(func=leaf_node_hitl)

  async def top_node(ctx: Context) -> str:
    return await ctx.run_node(middle_node)

  top_node = FunctionNode(func=top_node, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_nested_hitl',
      edges=[(START, top_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at leaf_node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req2',
              name='user_input',
              response={'text': 'World'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Check result
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_nested_hitl',
          {'node_name': 'leaf_node_hitl', 'output': {'text': 'World'}},
      ),
      (
          'test_agent_nested_hitl',
          {'node_name': 'middle_node', 'output': {'text': 'World'}},
      ),
      (
          'test_agent_nested_hitl',
          {'node_name': 'top_node', 'output': {'text': 'World'}},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_parallel_execution(request: pytest.FixtureRequest):
  """Tests a simple parent node running 3 parallel instances of dynamic node."""

  def echo_node(node_input: str) -> str:
    return node_input

  echo_node = FunctionNode(func=echo_node)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(echo_node, node_input=f'call_{i}') for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='dynamic_parallel',
      edges=[
          (START, parent_node),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  simplified = simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  )

  assert simplified[0] == (
      'dynamic_parallel',
      {'node_name': '__START__', 'output': user_event},
  ), 'Check parent output.'

  # Sort child events for deterministic assertion.
  child_events = sorted(simplified[1:-1], key=lambda x: x[1]['output'])
  assert child_events == [
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_0'},
      ),
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_1'},
      ),
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_2'},
      ),
  ]
  # And then assert the parent event separately.
  assert simplified[-1] == (
      'dynamic_parallel',
      {'node_name': 'parent_node', 'output': ['call_0', 'call_1', 'call_2']},
  )


@pytest.mark.asyncio
async def test_dynamic_node_parallel_mixed_hitl(request: pytest.FixtureRequest):
  """Tests mixed parallel execution with one HITL node and two simple nodes."""

  def simple_node(node_input: str) -> str:
    return f'simple_{node_input}'

  async def node_hitl() -> str:
    yield RequestInput(
        interrupt_id='req_hitl',
        message='request hitl',
        response_schema={'type': 'string'},
    )

  simple_node = FunctionNode(func=simple_node)
  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    t1 = ctx.run_node(simple_node, node_input='1')
    t2 = ctx.run_node(simple_node, node_input='2')
    t3 = ctx.run_node(node_hitl)
    return await asyncio.gather(t1, t2, t3)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_parallel_mixed',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause. simple_nodes should run.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  simplified1 = simplify_events_with_node(
      events1, map_dynamic_node_to_the_source=True
  )

  assert simplified1 == [
      (
          'test_agent_parallel_mixed',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'test_agent_parallel_mixed',
          {'node_name': 'simple_node', 'output': 'simple_1'},
      ),
      (
          'test_agent_parallel_mixed',
          {'node_name': 'simple_node', 'output': 'simple_2'},
      ),
      (
          'test_agent_parallel_mixed',
          types.Part(
              function_call=types.FunctionCall(
                  args={
                      'interrupt_id': 'req_hitl',
                      'message': 'request hitl',
                      'payload': None,
                      'response_schema': {'type': 'string'},
                  },
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
              )
          ),
      ),
  ], 'Check that simple nodes ran in the first run.'

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req_hitl',
              name='user_input',
              response={'text': 'HitlResponse'},
          )
      )
  )

  # Run 2: Resume. Simple nodes should NOT rerun.
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'test_agent_parallel_mixed',
          {'node_name': 'node_hitl', 'output': {'text': 'HitlResponse'}},
      ),
      (
          'test_agent_parallel_mixed',
          {
              'node_name': 'parent_node',
              'output': ['simple_1', 'simple_2', {'text': 'HitlResponse'}],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_parallel_hitl_all_resume(
    request: pytest.FixtureRequest,
):
  """Tests resuming multiple parallel HITL nodes at once."""

  async def node_hitl(node_input: str):
    yield RequestInput(
        interrupt_id=f'req_{node_input}',
        message=f'request {node_input}',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(node_hitl, node_input=str(i)) for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='parallel_hitl_all',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  parts = []
  for i in range(3):
    parts.append(
        types.Part(
            function_response=types.FunctionResponse(
                id=f'req_{i}',
                name='user_input',
                response={'text': f'resp_{i}'},
            )
        )
    )
  resume_payload = types.Content(role='user', parts=parts)

  # Run 2: Resume all.
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_0'}},
      ),
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_1'}},
      ),
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_2'}},
      ),
      (
          'parallel_hitl_all',
          {
              'node_name': 'parent_node',
              'output': [
                  {'text': 'resp_0'},
                  {'text': 'resp_1'},
                  {'text': 'resp_2'},
              ],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_parallel_hitl_partial_resume(
    request: pytest.FixtureRequest,
):
  """Tests resuming parallel HITL nodes in steps (partial resume)."""

  async def node_hitl(node_input: str):
    yield RequestInput(
        interrupt_id=f'req_{node_input}',
        message=f'request {node_input}',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(node_hitl, node_input=str(i)) for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='parallel_hitl_partial',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)
  invocation_id = events1[0].invocation_id

  # Resume 1: Respond to req_0 only.
  resume_payload1 = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req_0',
              name='user_input',
              response={'text': 'resp_0'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload1, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_0'}},
      ),
  ], 'Check that parent is NOT finished (not in events).'

  # Resume 2: Respond to req_1 and req_2.
  parts = []
  for i in [1, 2]:
    parts.append(
        types.Part(
            function_response=types.FunctionResponse(
                id=f'req_{i}',
                name='user_input',
                response={'text': f'resp_{i}'},
            )
        )
    )
  resume_payload2 = types.Content(role='user', parts=parts)

  events3 = await runner.run_async(
      new_message=resume_payload2, invocation_id=invocation_id
  )

  simplified3 = simplify_events_with_node(
      events3, map_dynamic_node_to_the_source=True
  )

  # Sort the events for deterministic assertion since parallel nodes may
  # finish in any order. We separate the parent event (last one)
  # from the child events.
  parent_event = simplified3[-1]
  child_events = sorted(simplified3[:-1], key=lambda x: x[1]['output']['text'])

  assert child_events == [
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_1'}},
      ),
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_2'}},
      ),
  ]
  assert parent_event == (
      'parallel_hitl_partial',
      {
          'node_name': 'parent_node',
          'output': [
              {'text': 'resp_0'},
              {'text': 'resp_1'},
              {'text': 'resp_2'},
          ],
      },
  )


@pytest.mark.asyncio
async def test_dynamic_node_with_multiple_events(
    request: pytest.FixtureRequest,
):  # pylint: disable=redefined-outer-name
  """Tests dynamic node execution with multiple events and a blocking sleep.

  Args:
    request: The pytest fixture request.

  Given:
    A workflow where 'node_parent' yields an event, then dynamically calls
    'node_dynamic', then yields a final event. 'node_dynamic' yields an
    event, performs a blocking time.sleep(), then yields a second event.
  When:
    The workflow agent is run asynchronously.
  Then:
    Events should be received in the order: parent-1, child-1, child-2,
    parent-2. Events yielded before time.sleep() should be received by the
    caller before sleep() is called, and the final output of 'node_parent'
    should contain both outputs of 'node_dynamic'.
  """
  sleep_started = False

  async def node_dynamic():
    nonlocal sleep_started

    yield 'child-1'
    sleep_started = True
    time.sleep(0.5)
    yield 'child-2'

  node_d = FunctionNode(func=node_dynamic)

  async def node_parent(ctx: Context):
    yield 'parent-1'
    output_d = await ctx.run_node(node_d)
    yield f'parent-2 with {output_d}'

  node_parent = FunctionNode(func=node_parent, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_dynamic_multi_events',
      edges=[
          (START, node_parent),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  events = []
  sleep_started_values = []
  async for e in agent.run_async(ctx):
    events.append(e)
    if simplify_event_with_node(e):
      sleep_started_values.append(sleep_started)

  simplified_events = simplify_events_with_node(events)

  # Check that sleep_started is False before child node goes to sleep.
  # This ensures that events before the sleep are indeed received by the outer
  # caller.
  assert sleep_started_values == [
      False,
      False,
      True,
      True,
  ], 'Check that sleep_started is False before child node goes to sleep.'

  # The event order should be P1, D1, D2, P2.

  # P1 event from node_parent
  agent_name, event_data = simplified_events[0]
  assert agent_name == 'test_workflow_agent_dynamic_multi_events'
  assert event_data['output'] == 'parent-1'
  assert event_data['node_name'] == 'node_parent'

  # D1 event from node_dynamic
  agent_name, event_data = simplified_events[1]
  assert agent_name == 'test_workflow_agent_dynamic_multi_events'
  assert event_data['output'] == 'child-1'
  assert event_data['node_name'].startswith('node_dynamic-')
  node_dynamic_name = event_data['node_name']

  # D2 event from node_dynamic
  agent_name, event_data = simplified_events[2]
  assert agent_name == 'test_workflow_agent_dynamic_multi_events'
  assert event_data['output'] == 'child-2'
  assert event_data['node_name'] == node_dynamic_name

  # P2 event from node_parent
  agent_name, event_data = simplified_events[3]
  assert agent_name == 'test_workflow_agent_dynamic_multi_events'
  assert event_data['output'] == "parent-2 with ['child-1', 'child-2']"
  assert event_data['node_name'] == 'node_parent'


@pytest.mark.asyncio
async def test_node_like_simple(request: pytest.FixtureRequest):
  """Tests passing functions directly to run_node."""

  def node_a() -> str:
    return 'A'

  def node_b() -> str:
    return 'B'

  async def node_c(ctx: Context) -> str:
    # Pass functions directly instead of FunctionNode objects
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C calls {output_a} & {output_b}'

  node_c = FunctionNode(func=node_c, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_node_like',
      edges=[
          (START, node_c),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_workflow_agent_node_like',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_a', 'output': 'A'},
      ),
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_b', 'output': 'B'},
      ),
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_c', 'output': 'C calls A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_node_like_nested(request: pytest.FixtureRequest):
  """Tests passing functions directly to run_node with nested calls."""

  def node_a() -> str:
    return 'A'

  def node_b() -> str:
    return 'B'

  async def func_c(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C -> {output_a} & {output_b}'

  node_c = FunctionNode(func=func_c, rerun_on_resume=True)

  async def func_d(ctx: Context) -> str:
    # Pass functions directly instead of FunctionNode objects
    output_c = await ctx.run_node(node_c)
    return f'D -> {output_c}'

  node_d = FunctionNode(func=func_d, rerun_on_resume=True)

  agent = Workflow(
      name='node_like_nested',
      edges=[
          (START, node_d),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'node_like_nested',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'node_like_nested',
          {'node_name': 'node_a', 'output': 'A'},
      ),
      (
          'node_like_nested',
          {'node_name': 'node_b', 'output': 'B'},
      ),
      (
          'node_like_nested',
          {'node_name': 'func_c', 'output': 'C -> A & B'},
      ),
      (
          'node_like_nested',
          {'node_name': 'func_d', 'output': 'D -> C -> A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_fails_if_caller_no_rerun(
    request: pytest.FixtureRequest,
):
  """Tests that dynamic execution fails if caller has rerun_on_resume=False."""

  def node_a() -> str:
    return 'A'

  node_a = FunctionNode(func=node_a)

  async def node_caller(ctx: Context) -> str:
    return await ctx.run_node(node_a)

  # Caller has rerun_on_resume=False (default is False for FunctionNode)
  node_caller = FunctionNode(func=node_caller, rerun_on_resume=False)

  agent = Workflow(
      name='test_agent_fail_no_rerun',
      edges=[(START, node_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start')

  with pytest.raises(ValueError, match='A node must have rerun_on_resume=True'):
    await runner.run_async(user_event)
