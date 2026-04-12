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

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow.execution_state import NodeState
from google.adk.workflow.execution_state import NodeStatus
from google.adk.workflow.node_runner import _check_and_schedule_nodes
from google.adk.workflow.node_runner import _cleanup_child_executions
from google.adk.workflow.node_runner import _execute_node
from google.adk.workflow.node_runner import _NodeCompletion
from google.adk.workflow.node_runner import _WorkflowRunState
from google.adk.workflow.trigger_processor import Trigger
from google.adk.workflow.workflow import WorkflowAgentState
from google.adk.workflow.workflow_graph import WorkflowGraph
from pydantic import ConfigDict
from pydantic import Field
import pytest
import pytest_asyncio
from typing_extensions import override

from . import testing_utils


class SimpleNode(BaseNode):
  """A simple node for testing that yields a single event."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  output_data: Any = None

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.output_data is not None:
      yield Event(data=self.output_data)


class MultiEventNode(BaseNode):
  """A node that yields multiple events for testing."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  events: list[Any] = Field(default_factory=list)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for event in self.events:
      yield event


class TestCleanupChildExecutions:
  """Tests for _cleanup_child_executions function."""

  def test_removes_child_executions(self):
    """Test that child executions are removed from agent state."""
    agent_state = WorkflowAgentState()
    parent_exec_id = 'parent-exec-123'

    # Add parent node
    agent_state.nodes['parent_node'] = NodeState(
        status=NodeStatus.RUNNING,
        execution_id=parent_exec_id,
    )

    # Add child nodes
    agent_state.nodes['child_node_1'] = NodeState(
        status=NodeStatus.COMPLETED,
        execution_id='child-exec-1',
        parent_execution_id=parent_exec_id,
    )
    agent_state.nodes['child_node_2'] = NodeState(
        status=NodeStatus.RUNNING,
        execution_id='child-exec-2',
        parent_execution_id=parent_exec_id,
    )

    # Add unrelated node (different parent)
    agent_state.nodes['unrelated_node'] = NodeState(
        status=NodeStatus.COMPLETED,
        execution_id='other-exec',
        parent_execution_id='other-parent',
    )

    _cleanup_child_executions(parent_exec_id, agent_state)

    # Child nodes should be removed
    assert 'child_node_1' not in agent_state.nodes
    assert 'child_node_2' not in agent_state.nodes
    # Parent and unrelated nodes should remain
    assert 'parent_node' in agent_state.nodes
    assert 'unrelated_node' in agent_state.nodes

  def test_no_children_to_remove(self):
    """Test when there are no child executions to remove."""
    agent_state = WorkflowAgentState()
    agent_state.nodes['node_1'] = NodeState(
        status=NodeStatus.COMPLETED,
        execution_id='exec-1',
    )

    _cleanup_child_executions('non-existent-parent', agent_state)

    # Node should remain unchanged
    assert 'node_1' in agent_state.nodes

  def test_empty_agent_state(self):
    """Test cleanup with empty agent state."""
    agent_state = WorkflowAgentState()

    # Should not raise any errors
    _cleanup_child_executions('any-parent-id', agent_state)

    assert len(agent_state.nodes) == 0


class TestCheckAndScheduleNodes:
  """Tests for _check_and_schedule_nodes function."""

  @pytest.fixture
  def simple_graph(self):
    """Create a simple graph with START -> node_a -> node_b."""
    node_a = SimpleNode(name='node_a')
    node_b = SimpleNode(name='node_b')
    return WorkflowGraph(
        edges=[
            Edge(from_node=START, to_node=node_a),
            Edge(from_node=node_a, to_node=node_b),
        ]
    )

  @pytest.fixture
  def nodes_map(self, simple_graph):
    """Create a nodes map from the graph."""
    return {node.name: node for node in simple_graph.nodes}

  @pytest_asyncio.fixture
  async def mock_run_state(self, simple_graph, nodes_map):
    """Create a mock WorkflowRunState for testing."""
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_invocation_context(agent)

    agent_state = WorkflowAgentState()

    return _WorkflowRunState(
        ctx=ctx,
        event_queue=asyncio.Queue(),
        graph=simple_graph,
        node_path='test_workflow',
        agent_state=agent_state,
        nodes_map=nodes_map,
        running_tasks={},
        dynamic_futures={},
        local_output_events=[],
        static_node_names=set(nodes_map.keys()),
        transfer_targets=[],
    )

  @pytest.mark.asyncio
  async def test_schedules_pending_nodes(self, mock_run_state):
    """Test that PENDING nodes are scheduled."""
    # Set node_a to PENDING
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.PENDING,
        input='test_input',
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should now have a running task
    assert 'node_a' in mock_run_state.running_tasks
    # Clean up the task
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_schedules_running_nodes_on_resume(self, mock_run_state):
    """Test that RUNNING nodes without tasks are scheduled (resume case)."""
    # Set node_a to RUNNING but not in running_tasks (simulating resume)
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.RUNNING,
        execution_id='exec-123',
        input='test_input',
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should now have a running task
    assert 'node_a' in mock_run_state.running_tasks
    # Clean up the task
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_does_not_reschedule_running_nodes_with_tasks(
      self, mock_run_state
  ):
    """Test that RUNNING nodes with existing tasks are not rescheduled."""
    # Set node_a to RUNNING with an existing task
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.RUNNING,
        execution_id='exec-123',
        input='test_input',
    )
    existing_task = mock.MagicMock()
    mock_run_state.running_tasks['node_a'] = existing_task

    _check_and_schedule_nodes(mock_run_state)

    # Task should remain the same
    assert mock_run_state.running_tasks['node_a'] is existing_task

  @pytest.mark.asyncio
  async def test_processes_trigger_buffer(self, mock_run_state):
    """Test that buffered triggers are processed and nodes scheduled."""
    # Add a trigger to the buffer
    mock_run_state.agent_state.trigger_buffer.setdefault('node_a', []).append(
        Trigger(input='buffered_input', triggered_by='__START__')
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should be scheduled
    assert 'node_a' in mock_run_state.running_tasks
    # Trigger buffer should be empty
    assert 'node_a' not in mock_run_state.agent_state.trigger_buffer
    # Node state should have the input from the trigger
    node_state = mock_run_state.agent_state.nodes['node_a']
    assert node_state.input == 'buffered_input'
    assert node_state.triggered_by == '__START__'
    # Status is RUNNING because _schedule_node sets it to RUNNING
    assert node_state.status == NodeStatus.RUNNING
    # Clean up
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_cleans_up_removed_dynamic_nodes(self, mock_run_state):
    """Test that dynamic nodes removed from state are cleaned from nodes_map."""
    # Add a dynamic node to nodes_map but not to agent_state.nodes
    dynamic_node = SimpleNode(name='dynamic_node')
    mock_run_state.nodes_map['dynamic_node'] = dynamic_node
    # It's not in static_node_names and not in agent_state.nodes

    _check_and_schedule_nodes(mock_run_state)

    # Dynamic node should be removed from nodes_map
    assert 'dynamic_node' not in mock_run_state.nodes_map


class TestExecuteNode:
  """Tests for _execute_node function."""

  @pytest_asyncio.fixture
  async def mock_ctx(self):
    """Create a mock InvocationContext."""
    agent = testing_utils.create_test_agent()
    return await testing_utils.create_invocation_context(agent)

  @pytest.fixture
  def mock_schedule_dynamic_node(self):
    """Create a mock schedule_dynamic_node function."""
    return mock.MagicMock(return_value=asyncio.Future())

  @pytest.mark.asyncio
  async def test_execute_node_yields_events(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node yields events from node.run."""
    node = SimpleNode(name='test_node', output_data={'key': 'value'})

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input='test_input',
        triggered_by='__START__',
        in_nodes={'__START__'},
        execution_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].data == {'key': 'value'}

  @pytest.mark.asyncio
  async def test_execute_node_assigns_author(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns author to events."""
    node = SimpleNode(name='test_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='exec-123',
        current_node_path='my_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].author == 'my_workflow'

  @pytest.mark.asyncio
  async def test_execute_node_assigns_node_name(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns node_name to events."""
    node = SimpleNode(name='my_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].node_name == 'my_node'

  @pytest.mark.asyncio
  async def test_execute_node_assigns_execution_id(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns execution_id to events."""
    node = SimpleNode(name='test_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='my-exec-id',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].execution_id == 'my-exec-id'

  @pytest.mark.asyncio
  async def test_execute_node_with_multiple_events(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node handles multiple events from a node."""
    node = MultiEventNode(
        name='multi_node',
        events=[
            Event(data='first'),
            Event(data='second'),
            Event(data='third'),
        ],
    )

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 3
    assert events[0].data == 'first'
    assert events[1].data == 'second'
    assert events[2].data == 'third'

  @pytest.mark.asyncio
  async def test_execute_node_converts_non_event_to_event(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that non-Event yields are wrapped in Event."""
    node = MultiEventNode(name='test_node', events=['raw_data'])

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], Event)
    assert events[0].data == 'raw_data'

  @pytest.mark.asyncio
  async def test_execute_node_skips_none_yields(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that None yields are skipped."""
    node = MultiEventNode(
        name='test_node', events=[None, Event(data='valid'), None]
    )

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        execution_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].data == 'valid'


class TestNodeCompletion:
  """Tests for _NodeCompletion dataclass."""

  def test_default_values(self):
    """Test default values of _NodeCompletion."""
    completion = _NodeCompletion(node_name='test_node')

    assert completion.node_name == 'test_node'
    assert completion.execution_id is None
    assert completion.node_interrupted is False
    assert completion.interrupt_ids == []
    assert completion.no_trigger is False
    assert completion.exception is None

  def test_with_interrupt(self):
    """Test _NodeCompletion with interrupt."""
    completion = _NodeCompletion(
        node_name='test_node',
        execution_id='exec-123',
        node_interrupted=True,
        interrupt_ids=['int-1', 'int-2'],
    )

    assert completion.node_interrupted is True
    assert completion.interrupt_ids == ['int-1', 'int-2']

  def test_with_exception(self):
    """Test _NodeCompletion with exception."""
    error = ValueError('test error')
    completion = _NodeCompletion(
        node_name='test_node',
        exception=error,
    )

    assert completion.exception is error
