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

"""Testings for Workflow retry logic on failures."""

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.retry_config import RetryConfig
from google.adk.workflow.workflow_graph import WorkflowGraph
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


class CustomRetryableError(Exception):
  """A custom error meant to be retried."""


class CustomNonRetryableError(Exception):
  """A custom error not meant to be retried."""


class _FlakyNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  message: str = Field(default='')
  succeed_on_iteration: int = Field(default=0)
  tracker: dict[str, Any] = Field(default_factory=dict)
  exception_to_raise: Exception = Field(...)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    iteration_count = self.tracker.get('iteration_count', 0) + 1
    self.tracker['iteration_count'] = iteration_count
    self.tracker.setdefault('retry_counts', []).append(ctx.retry_count)

    if iteration_count < self.succeed_on_iteration:
      raise self.exception_to_raise

    yield Event(
        data=self.message,
    )


@pytest.mark.asyncio
async def test_retry_on_matching_exception(request: pytest.FixtureRequest):
  """Tests that retries occur for exceptions listed in RetryConfig."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_no_retry_on_non_matching_exception(
    request: pytest.FixtureRequest,
):
  """Tests that no retry occurs for exceptions not listed in RetryConfig."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomNonRetryableError('Unexpected failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_no_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(CustomNonRetryableError, match='Unexpected failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_no_retry',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_retry_on_all_exceptions_if_not_specified(
    request: pytest.FixtureRequest,
):
  """Tests retries when `exceptions` is not specified.

  Retries should occur for any exception in this case.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=None,
      ),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_all',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_all',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_all',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_count_populated_correctly(
    request: pytest.FixtureRequest,
):
  """Tests that retry_count is populated correctly in the workflow context."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0, exceptions=['CustomRetryableError']
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_count_populated_correctly',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3
  assert flaky_node_in_agent.tracker['retry_counts'] == [0, 1, 2]


@pytest.mark.asyncio
async def test_retry_max_attempts_exceeded(
    request: pytest.FixtureRequest,
):
  """Tests that the agent stops retrying after exceeding `max_attempts`."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 4 times, but max_attempts is 3.
  # Total attempts = 3 (1 initial + 2 retries).
  # Attempt 1: retry_count = 0, fails.
  # Attempt 2: retry_count = 1, fails.
  # Attempt 3: retry_count = 2, fails. Now _should_retry_node returns False.
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=5,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Persisted failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          max_attempts=3,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_max_attempts',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(CustomRetryableError, match='Persisted failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_max_attempts',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_fails_without_retry_config(
    request: pytest.FixtureRequest,
):
  """Tests that the agent fails immediately if `retry_config` is None."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=None,
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_fails_without_retry_config',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(ValueError, match='Any failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_fails_without_retry_config',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_retries_with_empty_retry_config(
    request: pytest.FixtureRequest,
):
  """Tests that retries occur when `retry_config` is an empty instance."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Another failure'),
      retry_config=RetryConfig(),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retries_with_empty_retry_config',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retries_with_empty_retry_config',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retries_with_empty_retry_config',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_with_delay(request: pytest.FixtureRequest):
  """Tests retry with initial delay.

  This test verifies that the agent waits for the specified initial_delay before
  retrying a failed node.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Sleep test failure'),
      retry_config=RetryConfig(
          initial_delay=5.0,
          max_attempts=3,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_delay',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with mock.patch.object(
      asyncio, 'sleep', new_callable=mock.AsyncMock
  ) as mock_sleep:
    events = [e async for e in agent.run_async(ctx)]
    mock_sleep.assert_any_await(5.0)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_and_jitter(request: pytest.FixtureRequest):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=4,  # Fails 3 times
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Backoff test failure'),
      retry_config=RetryConfig(
          initial_delay=2.0,
          max_attempts=5,
          backoff_factor=3.0,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_backoff',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
    events = [e async for e in agent.run_async(ctx)]
    # Attempt 1: fails, delay = 2.0 * (3.0 ** 0) = 2.0
    # Attempt 2: fails, delay = 2.0 * (3.0 ** 1) = 6.0
    # Attempt 3: fails, delay = 2.0 * (3.0 ** 2) = 18.0
    mock_sleep.assert_has_awaits(
        [mock.call(2.0), mock.call(6.0), mock.call(18.0)]
    )

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 4


@pytest.mark.asyncio
async def test_retry_with_jitter(request: pytest.FixtureRequest):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Jitter test failure'),
      retry_config=RetryConfig(
          initial_delay=4.0,
          max_attempts=3,
          backoff_factor=1.0,
          jitter=0.5,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_jitter',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with (
      mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep,
      mock.patch('random.uniform', return_value=-1.0) as mock_random,
  ):
    events = [e async for e in agent.run_async(ctx)]

    # 4.0 + (-1.0) = 3.0
    mock_sleep.assert_any_await(3.0)
    # Called with -0.5 * 4.0, 0.5 * 4.0
    mock_random.assert_called_once_with(-2.0, 2.0)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2
