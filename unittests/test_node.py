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

"""Tests for @node decorator."""

from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.agent_node import AgentNode
from google.adk.workflow.llm_agent_node import LlmAgentNode
from google.adk.workflow.node import node
from google.adk.workflow.parallel_worker import ParallelWorker
from google.adk.workflow.retry_config import RetryConfig
from google.adk.workflow.tool_node import ToolNode
from google.adk.workflow.workflow import workflow_node_input
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node

ANY = mock.ANY


@pytest.mark.asyncio
async def test_node_decorator(request: pytest.FixtureRequest):
  """Tests that @node decorator can wrap a function and override its name."""

  @node(name="decorated_node")
  def my_func():
    return "Hello from decorated_func"

  assert my_func.name == "decorated_node"

  agent = Workflow(
      name="test_agent",
      edges=[
          (START, my_func),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          "test_agent",
          {
              "node_name": "decorated_node",
              "output": "Hello from decorated_func",
          },
      ),
  ]


def test_node_parallel_worker_instance():
  """Tests that node() can wrap a node in ParallelWorker."""

  @node(parallel_worker=True)
  def my_func(node_input):
    return node_input

  assert isinstance(my_func, ParallelWorker)
  assert my_func.name == "my_func"

  def other_func(x):
    return x

  parallel_node = node(other_func, parallel_worker=True)
  assert isinstance(parallel_node, ParallelWorker)
  assert parallel_node.name == "other_func"


@pytest.mark.asyncio
async def test_node_parallel_worker_execution(request: pytest.FixtureRequest):
  """Tests that a node with parallel_worker=True correctly processes inputs."""

  @node(parallel_worker=True)
  async def my_func(node_input):
    return node_input * 2

  agent = Workflow(
      name="test_agent",
      edges=[
          (START, my_func),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  token = workflow_node_input.set([1, 2, 3])
  try:
    events = [e async for e in agent.run_async(ctx)]
  finally:
    workflow_node_input.reset(token)

  # ParallelWorker returns a list of results.
  assert simplify_events_with_node(events) == [
      (
          "test_agent",
          {
              "node_name": "__START__",
              "output": [1, 2, 3],
          },
      ),
      (
          "test_agent",
          {
              "node_name": "my_func@0",
              "output": 2,
          },
      ),
      (
          "test_agent",
          {
              "node_name": "my_func@1",
              "output": 4,
          },
      ),
      (
          "test_agent",
          {
              "node_name": "my_func@2",
              "output": 6,
          },
      ),
      (
          "test_agent",
          {
              "node_name": "my_func",
              "output": [2, 4, 6],
          },
      ),
  ]


def test_node_decorator_rerun_on_resume():
  """Tests that @node decorator can override rerun_on_resume."""

  @node(name="decorated_node", rerun_on_resume=True)
  def my_func():
    return "Hello from decorated_func"

  assert isinstance(my_func, FunctionNode)
  assert my_func.rerun_on_resume

  @node()
  def my_func2():
    return "Hello from decorated_func2"

  assert isinstance(my_func2, FunctionNode)
  assert not my_func2.rerun_on_resume


def test_node_function_with_base_node():
  """Tests that node() function returns a copied node when given a BaseNode."""

  @node(name="original")
  def original():
    pass

  wrapped = node(original, name="overridden", rerun_on_resume=True)

  assert isinstance(wrapped, FunctionNode)
  assert wrapped is not original
  assert wrapped.name == "overridden"
  assert wrapped.rerun_on_resume


# BaseTool
class MyTool(BaseTool):
  name = "tool"
  description = "desc"

  async def _run_async_impl(self):
    return "done"


def test_node_no_unnecessary_wrap():
  """Tests that node() does not wrap LlmAgent, Agent, Tool, or func in OverridingNode."""

  # LlmAgent
  llm_agent = LlmAgent(name="llm")
  llm_node = node(llm_agent, name="overridden_llm")
  assert isinstance(llm_node, LlmAgentNode)
  assert llm_node.name == "overridden_llm"

  # BaseAgent
  agent = BaseAgent(name="agent")
  agent_node_inst = node(agent, name="overridden_agent", rerun_on_resume=True)
  assert isinstance(agent_node_inst, AgentNode)
  assert agent_node_inst.name == "overridden_agent"
  assert agent_node_inst.rerun_on_resume

  tool_inst = MyTool(name="tool", description="desc")
  t_node = node(tool_inst, name="overridden_tool")
  assert isinstance(t_node, ToolNode)
  assert t_node.name == "overridden_tool"

  # Callable
  def my_func():
    pass

  f_node = node(my_func, name="overridden_func", rerun_on_resume=True)
  assert isinstance(f_node, FunctionNode)
  assert f_node.name == "overridden_func"
  assert f_node.rerun_on_resume


class StatefulTool(BaseTool):
  """A tool that modifies state via tool_context."""

  async def run_async(self, *, args, tool_context):
    tool_context.state["tool_key"] = "tool_value"
    tool_context.state["tool_count"] = 10
    return {"status": "ok"}


class StatefulToolNoReturn(BaseTool):
  """A tool that modifies state but returns None."""

  async def run_async(self, *, args, tool_context):
    tool_context.state["silent_key"] = "silent_value"
    return None


@pytest.mark.asyncio
async def test_tool_node_state_delta(request: pytest.FixtureRequest):
  """Tests that state set via tool_context.state in ToolNode is persisted."""

  tool_node = ToolNode(
      tool=StatefulTool(name="stateful_tool", description="Sets state values"),
  )

  def read_state(tool_key: str, tool_count: int) -> str:
    return f"tool_key={tool_key}, tool_count={tool_count}"

  agent = Workflow(
      name="test_tool_node_state_delta",
      edges=[
          (START, tool_node),
          (tool_node, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          "test_tool_node_state_delta",
          {
              "node_name": "stateful_tool",
              "output": {"status": "ok"},
              "state_delta": {"tool_key": "tool_value", "tool_count": 10},
          },
      ),
      (
          "test_tool_node_state_delta",
          {
              "node_name": "read_state",
              "output": "tool_key=tool_value, tool_count=10",
          },
      ),
  ]


@pytest.mark.asyncio
async def test_tool_node_state_delta_no_return(
    request: pytest.FixtureRequest,
):
  """Tests that state is persisted even when tool returns None."""

  tool_node = ToolNode(
      tool=StatefulToolNoReturn(
          name="stateful_tool_no_return",
          description="Sets state, returns None",
      ),
  )

  def read_state(silent_key: str) -> str:
    return f"silent_key={silent_key}"

  agent = Workflow(
      name="test_tool_node_state_delta_no_return",
      edges=[
          (START, tool_node),
          (tool_node, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          "test_tool_node_state_delta_no_return",
          {
              "node_name": "stateful_tool_no_return",
              "output": None,
              "state_delta": {"silent_key": "silent_value"},
          },
      ),
      (
          "test_tool_node_state_delta_no_return",
          {
              "node_name": "read_state",
              "output": "silent_key=silent_value",
          },
      ),
  ]
