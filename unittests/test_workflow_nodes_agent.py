# Copyright 2025 Google LLC
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

"""Testings for the Workflow with agent nodes."""

import json
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import InvocationContext as BaseInvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event as AdkEvent
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.llm_agent_node import LlmAgentNode
from google.genai import types
from pydantic import BaseModel
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import simplify_events_with_node


class SimpleAgent(LlmAgent):
  """A simple agent for testing."""

  message: str = ''

  async def _run_async_impl(
      self, ctx: BaseInvocationContext
  ) -> AsyncGenerator[AdkEvent, None]:
    """Yields a single event with a message."""
    yield AdkEvent(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text=self.message)]),
    )


@pytest.mark.asyncio
async def test_run_async_with_agent_nodes(request: pytest.FixtureRequest):
  """Tests running a workflow with BaseAgent instances as nodes."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  agent_b = SimpleAgent(name='AgentB', message='World')
  agent = Workflow(
      name='wf_with_agents',
      edges=[
          (START, agent_a),
          (agent_a, agent_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('AgentA', 'Hello'),
      (
          'wf_with_agents',
          {
              'node_name': 'AgentA',
              'output': types.Content(parts=[types.Part(text='Hello')]),
          },
      ),
      ('AgentB', 'World'),
      (
          'wf_with_agents',
          {
              'node_name': 'AgentB',
              'output': types.Content(parts=[types.Part(text='World')]),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_agent_node_piping_data(
    request: pytest.FixtureRequest,
):
  """Tests that Event data from an agent node is piped to the next node."""
  agent_a = SimpleAgent(name='AgentA', message='Hello')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='wf_with_agent_piping',
      edges=[
          (START, agent_a),
          (agent_a, node_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  _ = [e async for e in agent.run_async(ctx)]

  assert node_b.received_inputs == [
      types.Content(parts=[types.Part(text='Hello')])
  ]


class MyOutput(BaseModel):
  name: str
  value: int


@pytest.mark.asyncio
async def test_run_with_output_schema():
  async def mock_run_async(*args, **kwargs):
    yield AdkEvent(
        content=types.Content(
            parts=[types.Part(text=json.dumps({'name': 'test', 'value': 123}))],
            role='model',
        ),
        author='test_agent',
    )

  with mock.patch.object(
      LlmAgent,
      'run_async',
      autospec=True,
      side_effect=mock_run_async,
  ):
    agent = LlmAgent(name='test_agent', output_schema=MyOutput)
    ctx = mock.create_autospec(Context, instance=True)
    ctx._invocation_context = mock.create_autospec(
        InvocationContext, instance=True
    )
    ctx._invocation_context.run_config = mock.MagicMock()
    ctx._invocation_context.run_config.response_modalities = ['text']
    node_input = 'some input'
    events = []
    agent_node = LlmAgentNode(agent=agent)
    async for event in agent_node.run(ctx=ctx, node_input=node_input):
      events.append(event)

  assert len(events) == 2
  assert isinstance(events[0], AdkEvent)
  assert isinstance(events[1], AdkEvent)
  assert events[1].data == {'name': 'test', 'value': 123}


@pytest.mark.asyncio
async def test_run_async_with_single_turn():
  async def mock_run_async(*args, **kwargs):
    yield AdkEvent(
        content=types.Content(parts=[types.Part(text='response')]),
        author='test_agent',
    )

  with mock.patch.object(
      LlmAgent,
      'run_async',
      autospec=True,
      side_effect=mock_run_async,
  ) as mock_agent_run:
    agent = LlmAgent(name='test_agent')
    invocation_ctx = await create_parent_invocation_context('test', agent)
    # Add some dummy events to the session to verify they are cleared
    invocation_ctx.session.events.append(
        AdkEvent(
            author='user',
            content=types.Content(parts=[types.Part(text='history')]),
        )
    )

    ctx = mock.create_autospec(Context, instance=True)
    ctx.get_invocation_context.return_value = invocation_ctx

    node_input = types.Content(parts=[types.Part(text='new input')])
    agent_node = LlmAgentNode(agent=agent, single_turn=True)
    async for _ in agent_node.run(ctx=ctx, node_input=node_input):
      pass

    # Verification
    # check that run_async was called
    mock_agent_run.assert_called()
    # Check the call arguments
    call_args = mock_agent_run.call_args
    call_kwargs = call_args.kwargs
    parent_context = call_kwargs['parent_context']

    # Verify session events contain only node_input and user_content is set
    assert len(parent_context.session.events) == 1
    assert parent_context.session.events[0].content == node_input
    # Verify other session attributes are preserved
    assert parent_context.session.id == invocation_ctx.session.id
    assert parent_context.session.app_name == invocation_ctx.session.app_name
    assert parent_context.session.user_id == invocation_ctx.session.user_id

    assert parent_context.user_content == node_input


@pytest.mark.asyncio
async def test_run_async_with_thought_signature():
  async def mock_run_async(*args, **kwargs):
    yield AdkEvent(
        content=types.Content(
            parts=[
                types.Part(
                    text='response',
                    thought_signature=b'secret',
                )
            ],
            role='model',
        ),
        author='test_agent',
    )

  with mock.patch.object(
      LlmAgent,
      'run_async',
      autospec=True,
      side_effect=mock_run_async,
  ):
    agent = LlmAgent(name='test_agent')
    ctx = mock.create_autospec(Context, instance=True)
    ctx._invocation_context = mock.create_autospec(
        InvocationContext, instance=True
    )
    ctx._invocation_context.run_config = mock.MagicMock()
    ctx._invocation_context.run_config.response_modalities = ['text']
    node_input = 'some input'

    agent_node = LlmAgentNode(agent=agent)
    events = [e async for e in agent_node.run(ctx=ctx, node_input=node_input)]

    assert len(events) == 2
    assert isinstance(events[1], AdkEvent)
    # This assertion expects the output event data to be a Content object
    # and thought_signature to be filtered out (set to None in Pydantic model).
    assert isinstance(events[1].data, types.Content)
    assert events[1].data.parts[0].thought_signature is None
