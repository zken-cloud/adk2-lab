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

"""Tests for WorkflowGraph validation."""

from google.adk.workflow import Edge
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow.workflow_graph import DEFAULT_ROUTE
from google.adk.workflow.workflow_graph import WorkflowGraph
import pytest

from .workflow_testing_utils import TestingNode


def test_valid_graph() -> None:
  """Tests that a valid graph passes validation."""
  node_a = TestingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_missing_start_node() -> None:
  """Tests that a graph missing the START node fails validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = WorkflowGraph(
      edges=[Edge(node_a, node_b)],
  )
  with pytest.raises(
      ValueError,
      match=(
          r"Graph validation failed\. START node \(name: '__START__'\) not"
          r' found in graph nodes\.'
      ),
  ):
    graph.validate_graph()


def test_unreachable_node() -> None:
  """Tests that a graph with an unreachable node fails validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')  # Unreachable
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_b, node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. The following nodes are unreachable'
          r' \(not a'
          r" to_node in any edge\): \['NodeB'\]"
      ),
  ):
    graph.validate_graph()


@pytest.mark.parametrize(
    'routes',
    [
        (None, None),
        ('route1', 'route1'),
        ('route1', 'route2'),
        ('route1', None),
    ],
)
def test_duplicate_edges_fail_validation(
    routes: tuple[str | None, str | None],
) -> None:
  """Tests that duplicate edges fail validation, regardless of routes."""
  node_a = TestingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(
              START,
              node_a,
              route=routes[0],
          ),
          Edge(
              START,
              node_a,
              route=routes[1],
          ),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. Duplicate edge found: from=__START__,'
          r' to=NodeA'
      ),
  ):
    graph.validate_graph()


def test_start_node_with_incoming_edge() -> None:
  """Tests graph with incoming edge to START node fails validation."""
  node_a = TestingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(node_a, START),
          Edge(START, node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. START node must not have incoming edges\.'
      ),
  ):
    graph.validate_graph()


def test_multiple_default_routes_fail_validation() -> None:
  """Tests that multiple DEFAULT_ROUTE edges from a node fail validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b, route=DEFAULT_ROUTE),
          Edge(node_a, node_c, route=DEFAULT_ROUTE),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r'Graph validation failed\. Multiple DEFAULT_ROUTE edges found from'
          r' node NodeA to NodeB and NodeC'
      ),
  ):
    graph.validate_graph()


def test_single_default_route_passes_validation() -> None:
  """Tests that a single DEFAULT_ROUTE edge from a node passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b, route=DEFAULT_ROUTE),
          Edge(node_a, node_c, route='another_route'),
      ],
  )
  graph.validate_graph()  # Should not raise


def test_duplicate_node_names_fail_validation() -> None:
  """Tests that duplicate nodes raise error."""

  node_a1 = TestingNode(name='NodeA')
  node_a2 = TestingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a1),
          Edge(node_a1, node_a2),
      ],
  )
  with pytest.raises(
      ValueError,
      match=(
          r"Graph validation failed\. Duplicate node names found: \['NodeA'\]\."
          r' This means multiple distinct node objects have the same name\. If'
          r' you intended to reuse the same node, ensure you pass the exact'
          r' same object instance\. If you intended to have distinct nodes,'
          r' ensure they have unique names\.'
      ),
  ):
    graph.validate_graph()


def test_from_edge_items_with_node_reuse_passes_validation() -> None:
  """Tests that node reuse with from_edge_items passes validation.

  The same my_node_func instance is used in the graph multiple times, and
  the workflow graph should recognize it as the same instance and not throw
  an error during validation.
  """

  def my_node_func() -> None:
    pass

  node_b = TestingNode(name='NodeB')
  graph = WorkflowGraph.from_edge_items([
      (START, my_node_func),
      (my_node_func, node_b),
  ])
  graph.validate_graph()  # Should not raise duplicate name error

  node_names = {n.name for n in graph.nodes}
  assert node_names == {'__START__', 'my_node_func', 'NodeB'}
  assert len(graph.nodes) == 3
  # Check that my_node_func was wrapped and deduplicated.
  func_node = next(n for n in graph.nodes if n.name == 'my_node_func')
  assert isinstance(func_node, FunctionNode)


def test_unconditional_cycle_fails_validation() -> None:
  """Tests that a cycle of unconditional edges (route=None) fails."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
          Edge(node_b, node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_unconditional_self_loop_fails_validation() -> None:
  """Tests that an unconditional self-loop (A -> A) fails."""
  node_a = TestingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_longer_unconditional_cycle_fails_validation() -> None:
  """Tests that a longer unconditional cycle (A -> B -> C -> A) fails."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
          Edge(node_b, node_c),
          Edge(node_c, node_a),
      ],
  )
  with pytest.raises(
      ValueError,
      match=r'Graph validation failed\. Unconditional cycle detected:',
  ):
    graph.validate_graph()


def test_conditional_cycle_passes_validation() -> None:
  """Tests that a cycle with a routed edge (loop pattern) passes."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
          Edge(node_b, node_a, route='retry'),
      ],
  )
  graph.validate_graph()  # Should not raise — routed back-edge


def test_conditional_self_loop_passes_validation() -> None:
  """Tests that a self-loop with a route passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_a, route='continue'),
          Edge(node_a, node_b, route='done'),
      ],
  )
  graph.validate_graph()  # Should not raise — routed self-loop


def test_dag_with_diamond_passes_validation() -> None:
  """Tests that a DAG with a diamond shape passes validation."""
  node_a = TestingNode(name='NodeA')
  node_b = TestingNode(name='NodeB')
  node_c = TestingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, node_c),
          Edge(node_b, node_c),
      ],
  )
  graph.validate_graph()  # Should not raise
