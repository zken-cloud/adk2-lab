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

"""Tests for agent_state_utils.reconstruct_state_from_events."""

from google.adk.events.event import Event
from google.adk.workflow.base_node import START
from google.adk.workflow.execution_state import NodeStatus
from google.adk.workflow.utils.agent_state_utils import reconstruct_state_from_events
from google.adk.workflow.workflow_graph import Edge
from google.adk.workflow.workflow_graph import WorkflowGraph
from google.genai import types
import pytest

from .workflow_testing_utils import TestingNode


def _make_graph(*node_names):
  """Creates a simple linear graph: START -> node1 -> node2 -> ..."""
  nodes = [TestingNode(name=n) for n in node_names]
  edges = [Edge(START, nodes[0])]
  for i in range(len(nodes) - 1):
    edges.append(Edge(nodes[i], nodes[i + 1]))
  return WorkflowGraph(edges=edges)


def test_returns_none_for_no_events():
  graph = _make_graph('A', 'B')
  result = reconstruct_state_from_events(
      session_events=[],
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is None


def test_returns_none_when_no_interrupted_nodes():
  """All nodes completed — no need to reconstruct."""
  graph = _make_graph('A', 'B')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='output_a',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b',
          data='output_b',
          invocation_id='inv-1',
          author='wf',
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is None


def test_reconstructs_interrupted_node():
  """Node A completed, node B interrupted."""
  graph = _make_graph('A', 'B')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='output_a',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert 'A' in result
  assert result['A'].status == NodeStatus.COMPLETED
  assert 'B' in result
  assert result['B'].status == NodeStatus.INTERRUPTED
  assert result['B'].interrupts == ['interrupt-1']
  assert result['B'].execution_id == 'exec-b'
  # B's input should be reconstructed from upstream A's data output.
  assert result['B'].input == 'output_a'


def test_resolved_interrupt_clears_node():
  """Node B was interrupted then resolved by user response."""
  graph = _make_graph('A', 'B')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='output_a',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
      # User resolves the interrupt
      Event(
          author='user',
          invocation_id='inv-1',
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='interrupt-1',
                          name='tool',
                          response={'result': 'done'},
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  # All interrupts resolved, no need to reconstruct
  assert result is None


def test_skips_current_invocation_events():
  """Events from current invocation should be ignored."""
  graph = _make_graph('A')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-2',  # Same as current
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is None


def test_dynamic_node_metadata_preserved():
  """Dynamic node metadata is captured from events."""
  graph = _make_graph('A')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='output_a',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/dyn-node-123',
          execution_id='exec-dyn',
          long_running_tool_ids={'interrupt-1'},
          source_node_name='my_dynamic_node',
          parent_execution_id='exec-a',
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert 'dyn-node-123' in result
  dyn_state = result['dyn-node-123']
  assert dyn_state.status == NodeStatus.INTERRUPTED
  assert dyn_state.source_node_name == 'my_dynamic_node'
  assert dyn_state.parent_execution_id == 'exec-a'


def test_ignores_events_from_other_workflows():
  """Events from nested or unrelated workflows are ignored."""
  graph = _make_graph('A')
  events = [
      # Event from a nested workflow (not direct child)
      Event(
          node_path='wf/nested_wf/A',
          execution_id='exec-a',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is None


def test_scopes_to_latest_interrupted_invocation():
  """Only events from the most recent interrupted invocation are used."""
  graph = _make_graph('A', 'B')
  events = [
      # Invocation 1: A and B both completed (previous full run).
      Event(
          node_path='wf/A',
          execution_id='exec-a-old',
          data='old_output_a',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b-old',
          data='old_output_b',
          invocation_id='inv-1',
          author='wf',
      ),
      # Invocation 2: A completed, B interrupted.
      Event(
          node_path='wf/A',
          execution_id='exec-a-new',
          data='new_output_a',
          invocation_id='inv-2',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b-new',
          long_running_tool_ids={'interrupt-2'},
          invocation_id='inv-2',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-2'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-3',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  # Should use inv-2 events only, not inv-1.
  assert result['A'].status == NodeStatus.COMPLETED
  assert result['A'].execution_id == 'exec-a-new'
  assert result['B'].status == NodeStatus.INTERRUPTED
  assert result['B'].execution_id == 'exec-b-new'


def test_start_node_always_present():
  """__START__ is always included as COMPLETED in reconstructed state."""
  graph = _make_graph('A')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert '__START__' in result
  assert result['__START__'].status == NodeStatus.COMPLETED


def test_interrupted_node_input_from_upstream_data():
  """Interrupted node's input is restored from upstream node's data."""
  graph = _make_graph('call_llm', 'execute_tools')
  call_llm_data = {'function_calls': [{'name': 'my_tool', 'args': {}}]}
  events = [
      Event(
          node_path='wf/call_llm',
          execution_id='exec-cl',
          data=call_llm_data,
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/execute_tools',
          execution_id='exec-et',
          long_running_tool_ids={'lr-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='my_tool', args={}, id='lr-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert result['execute_tools'].status == NodeStatus.INTERRUPTED
  assert result['execute_tools'].input == call_llm_data


def test_interrupted_node_input_multiple_data_events():
  """Upstream node with multiple data events produces list input."""
  graph = _make_graph('A', 'B')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='first',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          data='second',
          invocation_id='inv-1',
          author='wf',
      ),
      Event(
          node_path='wf/B',
          execution_id='exec-b',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert result['B'].input == ['first', 'second']


def test_interrupted_node_input_none_when_no_upstream_data():
  """Input stays None when upstream node has no data output."""
  graph = _make_graph('A')
  events = [
      Event(
          node_path='wf/A',
          execution_id='exec-a',
          long_running_tool_ids={'interrupt-1'},
          invocation_id='inv-1',
          author='wf',
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='interrupt-1'
                      )
                  )
              ]
          ),
      ),
  ]
  result = reconstruct_state_from_events(
      session_events=events,
      current_invocation_id='inv-2',
      workflow_path='wf',
      graph=graph,
  )
  assert result is not None
  assert result['A'].input is None
