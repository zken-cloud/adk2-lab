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

"""Testings for the Workflow HITL scenarios."""

import asyncio
import copy
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.execution_state import NodeStatus
from google.adk.workflow.utils.workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils.workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils.workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import testing_utils
from . import workflow_testing_utils
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import RequestInputNode

ANY = mock.ANY


class _TestingNode(BaseNode):
  """A node that produces a simple message."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  message: str = Field(default='')
  delay: float = Field(default=0)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.delay > 0:
      await asyncio.sleep(self.delay)
    yield Event(data=self.message)


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return 'Pending tool output'


@pytest.mark.asyncio
async def test_workflow_pause_and_resume(
    request: pytest.FixtureRequest,
):
  """Tests that a workflow can pause and resume.

  This test uses LlmAgent with LongRunningFunctionTool, which requires
  resumability to preserve the LLM's conversation state across interrupts.
  """
  node_a = _TestingNode(name='NodeA', message='Executing A')

  node_b = LlmAgent(
      name='NodeB_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tool'),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )
  node_c = _TestingNode(name='NodeC', message='Executing C')
  agent = Workflow(
      name='test_workflow_agent_hitl',
      edges=[
          (START, node_a),
          (node_a, node_b),
          (node_b, node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on the long-running function call.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  fc_event = workflow_testing_utils.find_function_call_event(
      events1, 'long_running_tool_func'
  )
  function_call_id = fc_event.content.parts[0].function_call.id

  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
          include_execution_id=True,
      )
  )

  expected_events1 = [
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {
                      'status': NodeStatus.RUNNING.value,
                      'execution_id': ANY,
                  }
              }
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'node_name': '__START__',
              'output': testing_utils.get_user_content('start workflow'),
              'execution_id': ANY,
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'execution_id': ANY,
                  },
              }
          },
      ),
      (
          'test_workflow_agent_hitl',
          {'node_name': 'NodeA', 'output': 'Executing A', 'execution_id': ANY},
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
                  'NodeB_agent': {
                      'status': NodeStatus.RUNNING.value,
                      'execution_id': ANY,
                  },
              }
          },
      ),
      (
          'NodeB_agent',
          types.Part.from_function_call(name='long_running_tool_func', args={}),
      ),
      (
          'NodeB_agent',
          types.Part.from_function_response(
              name='long_running_tool_func',
              response={'result': 'Pending tool output'},
          ),
      ),
      (
          'test_workflow_agent_hitl',
          {
              'node_name': 'NodeB_agent',
              'output': types.Content(
                  parts=[
                      types.Part.from_function_response(
                          name='long_running_tool_func',
                          response={'result': 'Pending tool output'},
                      )
                  ],
                  role='user',
              ),
              'execution_id': ANY,
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
                  'NodeB_agent': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [function_call_id],
                      'execution_id': ANY,
                  },
              },
          },
      ),
  ]
  assert simplified_events1 == expected_events1

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  # Resume with tool output.
  # In resumable mode, reuse the invocation_id so agent state is loaded.
  # In non-resumable mode, use a new invocation so state is reconstructed
  # from session events.
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
          include_execution_id=True,
      )
  )

  expected_events2 = [
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  START.name: {'status': NodeStatus.COMPLETED.value},
                  node_a.name: {'status': NodeStatus.COMPLETED.value},
                  node_b.name: {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {
                          function_call_id: {
                              'result': 'Final tool output',
                          }
                      },
                      'execution_id': ANY,
                  },
              },
          },
      ),
      ('NodeB_agent', 'LLM response after tool'),
      ('NodeB_agent', testing_utils.END_OF_AGENT),
      (
          'test_workflow_agent_hitl',
          {
              'node_name': 'NodeB_agent',
              'output': types.Content(
                  parts=[types.Part.from_text(text='LLM response after tool')],
                  role='model',
              ),
              'execution_id': ANY,
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
                  'NodeB_agent': {'status': NodeStatus.COMPLETED.value},
                  'NodeC': {
                      'status': NodeStatus.RUNNING.value,
                      'execution_id': ANY,
                  },
              }
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'node_name': 'NodeC',
              'output': 'Executing C',
              'execution_id': ANY,
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
                  'NodeB_agent': {'status': NodeStatus.COMPLETED.value},
                  'NodeC': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_workflow_agent_hitl', testing_utils.END_OF_AGENT),
  ]
  assert simplified_events2 == expected_events2


@pytest.mark.asyncio
async def test_workflow_interrupt_allows_parallel_execution(
    request: pytest.FixtureRequest,
):
  """Tests that if one node is interrupted, parallel nodes can execute.

  This test uses LlmAgent with LongRunningFunctionTool, which requires
  resumability to preserve the LLM's conversation state across interrupts.
  """
  node_a = LlmAgent(
      name='NodeA',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )
  node_b = _TestingNode(name='NodeB', message='Executing B', delay=0.5)
  agent = Workflow(
      name='test_workflow_agent_parallel_interrupt',
      edges=[
          (START, node_a),
          (START, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)
  fc_event = workflow_testing_utils.find_function_call_event(
      events, 'long_running_tool_func'
  )
  function_call_id = fc_event.content.parts[0].function_call.id

  simplified = workflow_testing_utils.simplify_events_with_node_and_agent_state(
      copy.deepcopy(events)
  )
  expected = [
      (
          'test_workflow_agent_parallel_interrupt',
          {'nodes': {'__START__': {'status': NodeStatus.RUNNING.value}}},
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {
              'node_name': '__START__',
              'output': testing_utils.get_user_content('start workflow'),
          },
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.RUNNING.value},
                  'NodeB': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'NodeA',
          types.Part.from_function_call(name='long_running_tool_func', args={}),
      ),
      (
          'NodeA',
          types.Part.from_function_response(
              name='long_running_tool_func',
              response={'result': 'Pending tool output'},
          ),
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {
              'node_name': 'NodeA',
              'output': types.Content(
                  parts=[
                      types.Part.from_function_response(
                          name='long_running_tool_func',
                          response={'result': 'Pending tool output'},
                      )
                  ],
                  role='user',
              ),
          },
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [function_call_id],
                  },
                  'NodeB': {'status': NodeStatus.RUNNING.value},
              },
          },
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {'node_name': 'NodeB', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_parallel_interrupt',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [function_call_id],
                  },
                  'NodeB': {'status': NodeStatus.COMPLETED.value},
              },
          },
      ),
  ]
  assert simplified == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_workflow_request_input_resume(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests resume with RequestInputEvent."""

  class UserDetails(BaseModel):
    name: str
    age: int

  node_a = RequestInputNode(
      name='NodeA_input',
      message='Please provide user details.',
      response_schema=UserDetails.model_json_schema(),
  )
  node_b = _TestingNode(name='NodeB', message='Received user details')
  agent = Workflow(
      name='test_workflow_agent_input_schema',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run and expect RequestInputEvent
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  request_input_event = workflow_testing_utils.find_function_call_event(
      events1, REQUEST_INPUT_FUNCTION_CALL_NAME
  )
  assert request_input_event is not None
  args = request_input_event.content.parts[0].function_call.args
  assert args['message'] == 'Please provide user details.'
  assert args['response_schema'] == {
      'properties': {
          'name': {'title': 'Name', 'type': 'string'},
          'age': {'title': 'Age', 'type': 'integer'},
      },
      'required': ['name', 'age'],
      'title': 'UserDetails',
      'type': 'object',
  }
  interrupt_id = get_request_input_interrupt_ids(request_input_event)[0]
  invocation_id = request_input_event.invocation_id

  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1)
      )
  )
  expected_events1 = [
      (
          'test_workflow_agent_input_schema',
          {'nodes': {'__START__': {'status': NodeStatus.RUNNING.value}}},
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'node_name': '__START__',
              'output': testing_utils.UserContent('start workflow'),
          },
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA_input': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_workflow_agent_input_schema',
          types.Part(
              function_call=types.FunctionCall(
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                  args={
                      'interrupt_id': interrupt_id,
                      'message': 'Please provide user details.',
                      'payload': None,
                      'response_schema': {
                          'properties': {
                              'name': {'title': 'Name', 'type': 'string'},
                              'age': {'title': 'Age', 'type': 'integer'},
                          },
                          'required': ['name', 'age'],
                          'title': 'UserDetails',
                          'type': 'object',
                      },
                  },
              )
          ),
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA_input': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id],
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events1 == expected_events1
  else:
    assert simplified_events1 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events1)
    )

  # Resume with user input
  user_input = create_request_input_response(
      interrupt_id, {'name': 'John', 'age': 30}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2)
      )
  )
  expected_events2 = [
      (
          'test_workflow_agent_input_schema',
          {'output': {'age': 30, 'name': 'John'}, 'node_name': 'NodeA_input'},
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA_input': {'status': NodeStatus.COMPLETED.value},
                  'NodeB': {
                      'status': NodeStatus.RUNNING.value,
                  },
              }
          },
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'node_name': 'NodeB',
              'output': 'Received user details',
          },
      ),
      (
          'test_workflow_agent_input_schema',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA_input': {'status': NodeStatus.COMPLETED.value},
                  'NodeB': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_workflow_agent_input_schema', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )


class _Request2InputsNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield RequestInput(message='input 1')
    yield RequestInput(message='input 2')


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_workflow_request_multiple_inputs_resume_one_by_one(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node requests multiple inputs and user provides responses one by one."""
  node_a = _Request2InputsNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests 2 inputs, invocation pauses.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1)
      )
  )

  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 2
  interrupt_id1 = get_request_input_interrupt_ids(req_events[0])[0]
  interrupt_id2 = get_request_input_interrupt_ids(req_events[1])[0]
  invocation_id = events1[0].invocation_id

  expected_events1 = [
      (
          'test_agent',
          {'nodes': {'__START__': {'status': NodeStatus.RUNNING.value}}},
      ),
      (
          'test_agent',
          {
              'node_name': START.name,
              'output': testing_utils.get_user_content('start'),
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'test_agent',
          testing_utils.simplify_content(req_events[1].content),
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id1, interrupt_id2],
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events1 == expected_events1
  else:
    assert simplified_events1 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events1)
    )

  # Run 2: provide 1st input, expect NodeOutputEvent, but node still interrupted
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {'text': 'response 1'})
      ),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2)
      )
  )
  expected_events2 = [
      (
          'test_agent',
          {'node_name': 'NodeA', 'output': {'text': 'response 1'}},
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id2],
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )

  # Run 3: provide 2nd input, expect NodeOutputEvent, node now completed
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id2, {'text': 'response 2'})
      ),
      invocation_id=invocation_id,
  )
  simplified_events3 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events3)
      )
  )
  expected_events3 = [
      (
          'test_agent',
          {'node_name': 'NodeA', 'output': {'text': 'response 2'}},
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events3 == expected_events3
  else:
    assert simplified_events3 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events3)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('resumable', [True, False])
async def test_workflow_request_multiple_inputs_resume_all_at_once(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node requests multiple inputs and user provides all inputs in one turn."""
  node_a = _Request2InputsNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests 2 inputs, invocation pauses.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1)
      )
  )

  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 2
  interrupt_id1 = get_request_input_interrupt_ids(req_events[0])[0]
  interrupt_id2 = get_request_input_interrupt_ids(req_events[1])[0]
  invocation_id = events1[0].invocation_id

  expected_events1 = [
      (
          'test_agent',
          {'nodes': {'__START__': {'status': NodeStatus.RUNNING.value}}},
      ),
      (
          'test_agent',
          {
              'node_name': START.name,
              'output': testing_utils.get_user_content('start'),
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'test_agent',
          testing_utils.simplify_content(req_events[1].content),
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id1, interrupt_id2],
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events1 == expected_events1
  else:
    assert simplified_events1 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events1)
    )

  # Run 2: provide both inputs, expect one NodeOutputEvent, node completed
  events2 = await runner.run_async(
      new_message=types.Content(
          parts=[
              create_request_input_response(
                  interrupt_id1, {'text': 'response 1'}
              ),
              create_request_input_response(
                  interrupt_id2, {'text': 'response 2'}
              ),
          ],
          role='user',
      ),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2)
      )
  )
  expected_events2 = [
      ('test_agent', {'node_name': 'NodeA', 'output': {'text': 'response 1'}}),
      ('test_agent', {'node_name': 'NodeA', 'output': {'text': 'response 2'}}),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    stripped = workflow_testing_utils.strip_checkpoint_events(expected_events2)
    # In non-resumable mode, interrupt resolution order may differ.
    assert sorted(simplified_events2, key=str) == sorted(stripped, key=str)


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_workflow_request_multiple_inputs_resume_all_and_verify_piped_output(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node requests multiple inputs and user provides all inputs in one turn, verifying downstream."""
  node_a = _Request2InputsNode(name='NodeA')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_agent',
      edges=[
          Edge(from_node=START, to_node=node_a),
          Edge(from_node=node_a, to_node=node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests 2 inputs
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 2
  interrupt_id1 = get_request_input_interrupt_ids(req_events[0])[0]
  interrupt_id2 = get_request_input_interrupt_ids(req_events[1])[0]
  invocation_id = events1[0].invocation_id

  # Run 2: provide both inputs
  await runner.run_async(
      new_message=types.Content(
          parts=[
              create_request_input_response(
                  interrupt_id1, {'text': 'response 1'}
              ),
              create_request_input_response(
                  interrupt_id2, {'text': 'response 2'}
              ),
          ],
          role='user',
      ),
      invocation_id=invocation_id,
  )

  # Check NodeB inputs
  # It should receive a list of the two outputs.
  assert len(node_b.received_inputs) == 1
  assert sorted(node_b.received_inputs[0], key=lambda x: x['text']) == [
      {'text': 'response 1'},
      {'text': 'response 2'},
  ]


class _YieldOutputAndRequestInputNode(BaseNode):
  """A node that yields output and requests input."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield Event(data='output 1')
    yield RequestInput(interrupt_id='req1')


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_workflow_yield_output_and_request_input(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node yields output and requests input, outputs are combined."""
  node_a = _YieldOutputAndRequestInputNode(name='NodeA')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node yields output and requests input
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  invocation_id = events1[0].invocation_id

  # Verify NodeA output event is present
  node_a_outputs = [
      e
      for e in events1
      if isinstance(e, Event) and e.node_name == 'NodeA' and e.data is not None
  ]
  assert len(node_a_outputs) == 1
  assert node_a_outputs[0].data == 'output 1'

  # Resume
  await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('req1', {'text': 'output 2'})
      ),
      invocation_id=invocation_id,
  )

  # The output order is deterministic because:
  # 1. 'output 1' is fetched from history.
  # 2. 'output 2' is collected as local event during resume.
  # _get_node_outputs uses _get_node_output_events which combines history +
  # local.
  # So 'output 1' should come before 'output 2'.
  assert node_b.received_inputs == [['output 1', {'text': 'output 2'}]]


class _RerunNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if 'count' not in ctx.session.state:
      ctx.session.state['count'] = 0

    approval = None
    if ctx.session.state['count'] == 0:
      if resume_input := ctx.resume_inputs.get('ask_approval'):
        ctx.session.state['count'] = 1
        approval = resume_input['approved']
      else:
        yield RequestInput(
            message='Needs approval', interrupt_id='ask_approval'
        )
        return
    yield Event(data={'approval': approval})


class _RerunNodeWithTwoInputs(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if resume_input := ctx.resume_inputs.get('req1'):
      yield Event(state={'input1': resume_input['text']})
    if resume_input := ctx.resume_inputs.get('req2'):
      yield Event(state={'input2': resume_input['text']})

    if 'input1' not in ctx.state and 'req1' not in ctx.resume_inputs:
      yield RequestInput(message='input 1', interrupt_id='req1')
      return

    if 'input2' not in ctx.state and 'req2' not in ctx.resume_inputs:
      yield RequestInput(message='input 2', interrupt_id='req2')
      return

    input1 = ctx.resume_inputs['req1']['text']
    input2 = ctx.resume_inputs['req2']['text']
    yield Event(
        data={
            'input1': input1,
            'input2': input2,
        },
    )


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_workflow_rerun_on_resume(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests node requests input and reruns itself upon resume."""
  node_a = _RerunNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests input
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
          include_execution_id=True,
      )
  )
  req_events = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id1 = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  if resumable:
    node_a_execution_id_1 = simplified_events1[-1][1]['nodes']['NodeA'][
        'execution_id'
    ]
    assert node_a_execution_id_1

    assert simplified_events1[-1] == (
        'test_agent',
        {
            'nodes': {
                '__START__': {'status': NodeStatus.COMPLETED.value},
                'NodeA': {
                    'status': NodeStatus.INTERRUPTED.value,
                    'interrupts': [interrupt_id1],
                    'execution_id': node_a_execution_id_1,
                },
            },
        },
    )
  else:
    node_a_execution_id_1 = ANY

  # Run 2: provide input, node reruns and completes
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {'approved': True})
      ),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
          include_execution_id=True,
      )
  )
  if resumable:
    # Verify execution_id stays the same even for rerun node
    node_a_execution_id_2 = simplified_events2[0][1]['nodes']['NodeA'][
        'execution_id'
    ]
    assert node_a_execution_id_1 == node_a_execution_id_2

  expected_events2 = [
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {interrupt_id1: {'approved': True}},
                      'execution_id': node_a_execution_id_1,
                  },
              }
          },
      ),
      (
          'test_agent',
          {
              'node_name': 'NodeA',
              'output': {'approval': True},
              'execution_id': node_a_execution_id_1,
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_workflow_rerun_with_multiple_inputs(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests node with rerun_on_resume=True requests multiple inputs and resumed one by one."""
  node_a = _RerunNodeWithTwoInputs(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests 1st input
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
          include_execution_id=True,
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 1
  interrupt_id1 = get_request_input_interrupt_ids(req_events1[0])[0]
  assert interrupt_id1 == 'req1'
  invocation_id = events1[0].invocation_id
  if resumable:
    node_a_execution_id_1 = simplified_events1[-1][1]['nodes']['NodeA'][
        'execution_id'
    ]
    assert node_a_execution_id_1

    assert simplified_events1[-1] == (
        'test_agent',
        {
            'nodes': {
                '__START__': {'status': NodeStatus.COMPLETED.value},
                'NodeA': {
                    'status': NodeStatus.INTERRUPTED.value,
                    'interrupts': [interrupt_id1],
                    'execution_id': node_a_execution_id_1,
                },
            },
        },
    )
  else:
    node_a_execution_id_1 = ANY

  # Run 2: provide 1st input, node reruns and requests 2nd input
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {'text': 'response 1'})
      ),
      invocation_id=invocation_id,
  )
  simplified_events2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
          include_execution_id=True,
      )
  )
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 1
  interrupt_id2 = get_request_input_interrupt_ids(req_events2[0])[0]
  assert interrupt_id2 == 'req2'
  if resumable:
    node_a_execution_id_2 = simplified_events2[0][1]['nodes']['NodeA'][
        'execution_id'
    ]
    assert node_a_execution_id_1 == node_a_execution_id_2

  expected_events2 = [
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {interrupt_id1: {'text': 'response 1'}},
                      'execution_id': node_a_execution_id_1,
                  },
              }
          },
      ),
      (
          'test_agent',
          types.Part(
              function_call=types.FunctionCall(
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                  args={
                      'interrupt_id': 'req2',
                      'message': 'input 2',
                      'payload': None,
                      'response_schema': None,
                  },
              )
          ),
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id2],
                      'resume_inputs': {interrupt_id1: {'text': 'response 1'}},
                      'execution_id': node_a_execution_id_1,
                  },
              },
          },
      ),
  ]
  if resumable:
    assert simplified_events2 == expected_events2
  else:
    assert simplified_events2 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events2)
    )

  # Run 3: provide 2nd input, node reruns and completes
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id2, {'text': 'response 2'})
      ),
      invocation_id=invocation_id,
  )
  simplified_events3 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events3),
          include_resume_inputs=True,
          include_execution_id=True,
      )
  )
  if resumable:
    node_a_execution_id_3 = simplified_events3[0][1]['nodes']['NodeA'][
        'execution_id'
    ]
    assert node_a_execution_id_1 == node_a_execution_id_3

  expected_events3 = [
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {
                      'status': NodeStatus.RUNNING.value,
                      'resume_inputs': {
                          interrupt_id1: {'text': 'response 1'},
                          interrupt_id2: {'text': 'response 2'},
                      },
                      'execution_id': node_a_execution_id_1,
                  },
              }
          },
      ),
      (
          'test_agent',
          {
              'node_name': 'NodeA',
              'output': {'input1': 'response 1', 'input2': 'response 2'},
              'execution_id': node_a_execution_id_1,
          },
      ),
      (
          'test_agent',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'NodeA': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_agent', testing_utils.END_OF_AGENT),
  ]
  if resumable:
    assert simplified_events3 == expected_events3
  else:
    assert simplified_events3 == (
        workflow_testing_utils.strip_checkpoint_events(expected_events3)
    )


class _MultiHitlRerunNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if not ctx.resume_inputs.get('req1'):
      yield Event(data='output1')
      yield RequestInput(interrupt_id='req1', message='request 1')
      return
    if not ctx.resume_inputs.get('req2'):
      yield Event(data='output2')
      yield RequestInput(interrupt_id='req2', message='request 2')
      return
    yield Event(data='output3')


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_rerun_with_multiple_hitl_and_outputs(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests that a re-runnable node with multiple HITL accumulates outputs."""
  node_a = _MultiHitlRerunNode(name='NodeA')
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_agent_multi_hitl',
      edges=[
          (START, node_a),
          (node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()
  runner1 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  runner2 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  runner3 = Runner(
      app=app,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
  session = await session_service.create_session(
      app_name=app.name, user_id='test_user'
  )

  async def collect_events(agen):
    events = []
    async for e in agen:
      events.append(e)
    return events

  # Run 1: node yields output1 and requests input1
  events1 = await collect_events(
      runner1.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content('start'),
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 1
  assert get_request_input_interrupt_ids(req_events1[0])[0] == 'req1'
  invocation_id = events1[0].invocation_id

  # Run 2: provide input1, node yields output2 and requests input2.
  events2 = await collect_events(
      runner2.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.UserContent(
              create_request_input_response('req1', {'text': 'response 1'})
          ),
          invocation_id=invocation_id if resumable else None,
      )
  )
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 1
  assert get_request_input_interrupt_ids(req_events2[0])[0] == 'req2'

  # Run 3 with new runner: provide input2, node yields output3 and completes
  await collect_events(
      runner3.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.UserContent(
              create_request_input_response('req2', {'text': 'response 2'})
          ),
          invocation_id=invocation_id if resumable else None,
      )
  )

  assert node_b.received_inputs == [[
      'output1',
      'output2',
      'output3',
  ]]


class _SimultaneousInputsNode(BaseNode):
  """A node that requests multiple inputs simultaneously."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if not ctx.resume_inputs:
      # First run: request both inputs simultaneously.
      yield RequestInput(interrupt_id='req1', message='input 1')
      yield RequestInput(interrupt_id='req2', message='input 2')
      return

    # All inputs should be available when we rerun.
    yield Event(
        data={
            'input1': ctx.resume_inputs['req1']['text'],
            'input2': ctx.resume_inputs['req2']['text'],
        },
    )


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_rerun_on_resume_waits_for_all_interrupts(
    request: pytest.FixtureRequest,
    resumable: bool,
):
  """Tests that a rerun_on_resume node is not rerun until all pending interrupts are resolved."""
  node_a = _SimultaneousInputsNode(name='NodeA')
  agent = Workflow(
      name='test_agent',
      edges=[Edge(from_node=START, to_node=node_a)],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: node requests both inputs simultaneously.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  simplified1 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events1),
          include_resume_inputs=True,
      )
  )
  req_events1 = workflow_testing_utils.get_request_input_events(events1)
  assert len(req_events1) == 2
  interrupt_ids = []
  for e in req_events1:
    interrupt_ids.extend(get_request_input_interrupt_ids(e))
  assert set(interrupt_ids) == {'req1', 'req2'}
  invocation_id = events1[0].invocation_id

  # Final checkpoint should show INTERRUPTED with both interrupt_ids.
  if resumable:
    final_state1 = simplified1[-1][1]
    assert final_state1['nodes']['NodeA']['status'] == (
        NodeStatus.INTERRUPTED.value
    )
    assert set(final_state1['nodes']['NodeA']['interrupts']) == {
        'req1',
        'req2',
    }

  # Run 2: provide only req1 — node should stay INTERRUPTED, NOT rerun.
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('req1', {'text': 'response 1'})
      ),
      invocation_id=invocation_id,
  )
  simplified2 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events2),
          include_resume_inputs=True,
      )
  )

  # Node should remain INTERRUPTED with req2 still pending.
  # resume_inputs should accumulate req1's response.
  if resumable:
    final_state2 = simplified2[-1][1]
    assert final_state2['nodes']['NodeA']['status'] == (
        NodeStatus.INTERRUPTED.value
    )
    assert final_state2['nodes']['NodeA']['interrupts'] == ['req2']
    assert final_state2['nodes']['NodeA']['resume_inputs'] == {
        'req1': {'text': 'response 1'},
    }

  # The node should NOT have produced any RequestInput or data output.
  req_events2 = workflow_testing_utils.get_request_input_events(events2)
  assert len(req_events2) == 0

  # Run 3: provide req2 — now all interrupts resolved, node should rerun.
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response('req2', {'text': 'response 2'})
      ),
      invocation_id=invocation_id,
  )
  simplified3 = (
      workflow_testing_utils.simplify_events_with_node_and_agent_state(
          copy.deepcopy(events3),
          include_resume_inputs=True,
      )
  )

  # Node should have rerun and completed with both responses.
  # Last event is END_OF_AGENT, second-to-last is the final agent state.
  if resumable:
    final_state3 = simplified3[-2][1]
    assert final_state3['nodes']['NodeA']['status'] == (
        NodeStatus.COMPLETED.value
    )

  # Check the node produced the expected output.
  data_events = [
      e
      for e in events3
      if hasattr(e, 'data') and e.data is not None and isinstance(e.data, dict)
  ]
  assert len(data_events) == 1
  assert data_events[0].data == {
      'input1': 'response 1',
      'input2': 'response 2',
  }
