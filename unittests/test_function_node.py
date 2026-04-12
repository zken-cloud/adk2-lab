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

"""Testings for the FunctionNode."""

import copy
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Optional
from typing import Union
from unittest import mock

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event import Event as AdkEvent
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow.execution_state import NodeStatus
from google.adk.workflow.utils.workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils.workflow_hitl_utils import get_request_input_interrupt_ids
from google.genai import types
from pydantic import BaseModel
import pytest

from . import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import get_request_input_events
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import simplify_events_with_node_and_agent_state

ANY = mock.ANY


@pytest.mark.asyncio
async def test_various_function_nodes(request: pytest.FixtureRequest):
  """Tests that Workflow can run with various function nodes."""

  async def async_gen_func(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        data='Hello from AsyncGen',
    )

  def sync_func_out(ctx: Context) -> str:
    return 'Hello from SyncFunc'

  async def async_func_out(ctx: Context) -> str:
    return 'Hello from AsyncFunc'

  def sync_func_no_out(ctx: Context) -> None:
    return None

  async def async_func_no_out(ctx: Context) -> None:
    return None

  def sync_gen_func(
      ctx: Context,
  ) -> Generator[Any, None, None]:
    yield Event(
        data='Hello from SyncGen',
    )

  async def async_gen_func_raw_output(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield 'Hello from AsyncGenRawOutput'

  def sync_gen_func_raw_output(
      ctx: Context,
  ) -> Generator[Any, None, None]:
    yield 'Hello from SyncGenRawOutput'

  agent = Workflow(
      name='test_workflow_agent_various_function_nodes',
      edges=[
          (START, async_gen_func),
          (async_gen_func, sync_func_out),
          (sync_func_out, async_func_out),
          (async_func_out, sync_func_no_out),
          (sync_func_no_out, async_func_no_out),
          (async_func_no_out, sync_gen_func),
          (sync_gen_func, async_gen_func_raw_output),
          (async_gen_func_raw_output, sync_gen_func_raw_output),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  # Functions with no output (sync_func_no_out, async_func_no_out)
  # will not produce events.
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_various_function_nodes',
          {'node_name': 'async_gen_func', 'output': 'Hello from AsyncGen'},
      ),
      (
          'test_workflow_agent_various_function_nodes',
          {'node_name': 'sync_func_out', 'output': 'Hello from SyncFunc'},
      ),
      (
          'test_workflow_agent_various_function_nodes',
          {'node_name': 'async_func_out', 'output': 'Hello from AsyncFunc'},
      ),
      (
          'test_workflow_agent_various_function_nodes',
          {'node_name': 'sync_gen_func', 'output': 'Hello from SyncGen'},
      ),
      (
          'test_workflow_agent_various_function_nodes',
          {
              'node_name': 'async_gen_func_raw_output',
              'output': 'Hello from AsyncGenRawOutput',
          },
      ),
      (
          'test_workflow_agent_various_function_nodes',
          {
              'node_name': 'sync_gen_func_raw_output',
              'output': 'Hello from SyncGenRawOutput',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection(request: pytest.FixtureRequest):
  """Tests that FunctionNode can inject parameters from workflow state."""

  async def set_state_node_fn(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'param1': 'value1'},
    )

  def check_state_node_fn(param1: str, param2: str = 'default2') -> str:
    return f'param1={param1}, param2={param2}'

  agent = Workflow(
      name='test_workflow_agent_state_injection',
      edges=[
          (START, set_state_node_fn),
          (set_state_node_fn, check_state_node_fn),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_state_injection',
          {
              'node_name': 'check_state_node_fn',
              'output': 'param1=value1, param2=default2',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection_missing_param(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode raises error for missing param."""

  def check_state_node_fn(param1: str) -> str:
    return f'param1={param1}'

  agent = Workflow(
      name='test_workflow_agent_state_injection_missing',
      edges=[
          (START, check_state_node_fn),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  with pytest.raises(ValueError, match='Missing value for parameter "param1"'):
    async for _ in agent.run_async(ctx):
      pass


@pytest.mark.asyncio
async def test_function_node_type_checking(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode performs type checking."""

  async def set_state_node_fn(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'p1': 'a string'},
    )

  def check_type_node_fn(p1: int) -> str:
    return f'p1={p1}'

  agent = Workflow(
      name='test_type_checking',
      edges=[
          (START, set_state_node_fn),
          (set_state_node_fn, check_type_node_fn),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  with pytest.raises(
      TypeError,
      match=(
          'Parameter "p1" of function "check_type_node_fn" expects type '
          "<class 'int'> but received type <class 'str'>."
      ),
  ):
    async for _ in agent.run_async(ctx):
      pass


@pytest.mark.asyncio
async def test_function_node_input_injection(request: pytest.FixtureRequest):
  """Tests that FunctionNode can inject parameters from node_input."""

  def node1_fn() -> dict[str, Any]:
    return {'p1': 'value1_from_node_input', 'p2': 100}

  def node2_fn(node_input: dict[str, Any]) -> str:
    return f"p1={node_input['p1']}, p2={node_input['p2']}"

  agent = Workflow(
      name='test_workflow_agent_input_injection_dict',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_dict', agent
  )
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_injection_dict',
          {
              'node_name': 'node1_fn',
              'output': {'p1': 'value1_from_node_input', 'p2': 100},
          },
      ),
      (
          'test_workflow_agent_input_injection_dict',
          {
              'node_name': 'node2_fn',
              'output': 'p1=value1_from_node_input, p2=100',
          },
      ),
  ]


class MyModel(BaseModel):
  p1: str
  p2: int


@pytest.mark.asyncio
async def test_function_node_input_injection_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode can inject dict as pydantic model into node_input."""

  def node1_fn() -> dict[str, Any]:
    return {'p1': 'value1_from_node_input', 'p2': 100}

  def node2_fn(node_input: MyModel) -> str:
    return f'p1={node_input.p1}, p2={node_input.p2}'

  agent = Workflow(
      name='test_workflow_agent_input_injection_pydantic',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_pydantic', agent
  )
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_injection_pydantic',
          {
              'node_name': 'node1_fn',
              'output': {'p1': 'value1_from_node_input', 'p2': 100},
          },
      ),
      (
          'test_workflow_agent_input_injection_pydantic',
          {
              'node_name': 'node2_fn',
              'output': 'p1=value1_from_node_input, p2=100',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_input_list_wrong_type(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode raises TypeError for list[T] with non-list input."""

  def node1_fn() -> int:
    return 123

  def node2_fn(node_input: list[MyModel]) -> str:
    return f'p1={node_input[0].p1}'

  agent = Workflow(
      name='test_workflow_agent_input_list_wrong_type',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_pydantic', agent
  )
  with pytest.raises(
      TypeError,
  ):
    async for _ in agent.run_async(ctx):
      pass


@pytest.mark.asyncio
async def test_function_node_input_list_no_item_type(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode handles list without item type."""

  def node1_fn() -> list[int]:
    return [1, 2]

  def node2_fn(node_input: list) -> str:
    return f'list={node_input}'

  agent = Workflow(
      name='test_workflow_agent_input_list_no_item_type',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_pydantic', agent
  )
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_list_no_item_type',
          {
              'node_name': 'node1_fn',
              'output': [1, 2],
          },
      ),
      (
          'test_workflow_agent_input_list_no_item_type',
          {
              'node_name': 'node2_fn',
              'output': 'list=[1, 2]',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_input_and_state_injection(
    request: pytest.FixtureRequest,
):
  """Tests parameter injection from node_input and state."""

  async def nodea_fn(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'p_state': 'value_from_state'},
    )
    yield 'value_A'

  def nodeb_fn(
      node_input: str, p_state: str, p_default: str = 'default2'
  ) -> str:
    return f'node_input={node_input}, p_state={p_state}, p_default={p_default}'

  agent = Workflow(
      name='test_node_param_injection_single_and_state',
      edges=[
          (START, nodea_fn),
          (nodea_fn, nodeb_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_single_and_state', agent
  )
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_node_param_injection_single_and_state',
          {'node_name': 'nodea_fn', 'output': 'value_A'},
      ),
      (
          'test_node_param_injection_single_and_state',
          {
              'node_name': 'nodeb_fn',
              'output': (
                  'node_input=value_A, p_state=value_from_state,'
                  ' p_default=default2'
              ),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode can inject dict from state as pydantic model."""

  async def node1_fn(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'my_model': {'p1': 'value1_from_state', 'p2': 200}},
    )

  def node2_fn(my_model: MyModel) -> str:
    return f'p1={my_model.p1}, p2={my_model.p2}'

  agent = Workflow(
      name='test_workflow_agent_state_injection_pydantic',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__ + '_pydantic', agent
  )
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_state_injection_pydantic',
          {
              'node_name': 'node2_fn',
              'output': 'p1=value1_from_state, p2=200',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_hitl(request: pytest.FixtureRequest):
  """Tests that FunctionNode can trigger HITL."""

  def request_input_fn() -> Generator[Any, None, None]:
    yield RequestInput(message='Provide input')

  def process_input_fn(node_input: dict[str, Any]) -> str:
    return f"received: {node_input['text']}"

  agent = Workflow(
      name='test_workflow_agent_hitl',
      edges=[
          (START, request_input_fn),
          (request_input_fn, process_input_fn),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on RequestInput.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  req_events = get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  simplified_events1 = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events1)
  )
  assert simplified_events1 == [
      (
          'test_workflow_agent_hitl',
          {'nodes': {'__START__': {'status': NodeStatus.RUNNING.value}}},
      ),
      (
          'test_workflow_agent_hitl',
          {'node_name': '__START__', 'output': user_event},
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'request_input_fn': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_workflow_agent_hitl',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'request_input_fn': {
                      'status': NodeStatus.INTERRUPTED.value,
                      'interrupts': [interrupt_id],
                  },
              }
          },
      ),
  ]

  # Resume with user input
  user_input = create_request_input_response(
      interrupt_id, {'text': 'Hello from user'}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )
  simplified_events2 = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events2)
  )
  assert simplified_events2 == [
      (
          'test_workflow_agent_hitl',
          {
              'node_name': 'request_input_fn',
              'output': {'text': 'Hello from user'},
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'request_input_fn': {'status': NodeStatus.COMPLETED.value},
                  'process_input_fn': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'node_name': 'process_input_fn',
              'output': 'received: Hello from user',
          },
      ),
      (
          'test_workflow_agent_hitl',
          {
              'nodes': {
                  '__START__': {'status': NodeStatus.COMPLETED.value},
                  'request_input_fn': {'status': NodeStatus.COMPLETED.value},
                  'process_input_fn': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('test_workflow_agent_hitl', testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_function_node_adk_events(request: pytest.FixtureRequest):
  """Tests that FunctionNode can emit ADK events."""

  def adk_events_fn() -> Generator[Any, None, None]:
    yield AdkEvent(
        author='some_agent', content=types.Content(parts=[{'text': 'event 1'}])
    )
    yield AdkEvent(
        author='some_agent', content=types.Content(parts=[{'text': 'event 2'}])
    )

  agent = Workflow(
      name='test_workflow_agent_adk_events',
      edges=[
          (START, adk_events_fn),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert len(events) == 2
  assert isinstance(events[0], AdkEvent)
  assert events[0].content.parts[0].text == 'event 1'
  assert isinstance(events[1], AdkEvent)
  assert events[1].content.parts[0].text == 'event 2'


@pytest.mark.asyncio
async def test_function_node_list_conversion_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode correctly converts list[dict] to list[BaseModel]."""

  class Section(BaseModel):
    section_name: str
    content: str

  async def upstream_func(ctx: Context) -> list[dict[str, str]]:
    return [
        {'section_name': 's1', 'content': 'c1'},
        {'section_name': 's2', 'content': 'c2'},
    ]

  received_input = None

  async def aggregate(node_input: list[Section]) -> str:
    nonlocal received_input
    received_input = node_input
    return 'Done'

  agent = Workflow(
      name='test_function_node_list_conversion_pydantic',
      edges=[
          (START, upstream_func),
          (upstream_func, aggregate),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  unused_events = [e async for e in agent.run_async(ctx)]

  # Check if received_input contains Section objects
  assert isinstance(received_input, list)
  assert len(received_input) == 2
  assert isinstance(received_input[0], Section)
  assert received_input[0].section_name == 's1'


@pytest.mark.asyncio
async def test_function_node_dict_conversion_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode correctly converts dict[str, dict] to dict[str, BaseModel]."""

  class Section(BaseModel):
    section_name: str
    content: str

  async def upstream_func(ctx: Context) -> dict[str, dict[str, str]]:
    return {
        'one': {'section_name': 's1', 'content': 'c1'},
        'two': {'section_name': 's2', 'content': 'c2'},
    }

  received_input = None

  async def aggregate(node_input: dict[str, Section]) -> str:
    nonlocal received_input
    received_input = node_input
    return 'Done'

  agent = Workflow(
      name='test_function_node_dict_conversion_pydantic',
      edges=[
          (START, upstream_func),
          (upstream_func, aggregate),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  unused_events = [e async for e in agent.run_async(ctx)]

  # Check if received_input contains Section objects
  assert isinstance(received_input, dict)
  assert len(received_input) == 2
  assert isinstance(received_input['one'], Section)
  assert received_input['one'].section_name == 's1'
  assert isinstance(received_input['two'], Section)
  assert received_input['two'].content == 'c2'


@pytest.mark.asyncio
async def test_function_node_no_data_returns_none(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode returns Event with data=None if function returns only control event."""

  def func_no_data() -> Event:
    return Event(route='some_route')

  agent = Workflow(
      name='test_function_node_no_data_returns_none',
      edges=[
          (START, func_no_data),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert len(events) == 1
  assert events[0].data is None
  assert events[0].actions.route == 'some_route'


@pytest.mark.asyncio
async def test_function_node_yield_content(
    request: pytest.FixtureRequest,
):
  """Tests that yielding types.Content sets data=None and content=Content."""

  def func_yield_content() -> Generator[Any, None, None]:
    yield types.Content(parts=[types.Part(text='some content')])

  agent = Workflow(
      name='test_function_node_yield_content',
      edges=[
          (START, func_yield_content),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert len(events) == 1
  assert events[0].data is None
  assert events[0].content is not None
  assert events[0].content.parts[0].text == 'some content'


@pytest.mark.asyncio
async def test_function_node_yield_event_with_content(
    request: pytest.FixtureRequest,
):
  """Tests that yielding Event(content=...) retains data=None and content=... ."""

  def func_yield_event_with_content() -> Generator[Any, None, None]:
    yield Event(content=types.Content(parts=[types.Part(text='some content')]))

  agent = Workflow(
      name='test_function_node_yield_event_with_content',
      edges=[
          (START, func_yield_event_with_content),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert len(events) == 1
  assert events[0].data is None
  assert events[0].content is not None
  assert events[0].content.parts[0].text == 'some content'


@pytest.mark.asyncio
async def test_content_to_str_auto_conversion(
    request: pytest.FixtureRequest,
):
  """Tests that Content is auto-converted to str when function expects str."""
  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return f'Hello, {node_input}!'

  agent = Workflow(
      name='test_content_to_str',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content('start workflow')
  await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert isinstance(received_inputs[0], str)
  assert received_inputs[0] == 'start workflow'


@pytest.mark.asyncio
async def test_content_to_str_multi_part(
    request: pytest.FixtureRequest,
):
  """Tests Content with multiple text parts is concatenated."""
  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return node_input

  agent = Workflow(
      name='test_content_to_str_multi_part',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content(
      types.Content(
          parts=[
              types.Part(text='Hello '),
              types.Part(text='World'),
          ],
          role='user',
      )
  )
  await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'Hello World'


@pytest.mark.asyncio
async def test_content_to_str_warns_on_non_text(
    request: pytest.FixtureRequest,
    caplog,
):
  """Tests that non-text parts produce a warning during conversion."""
  import logging

  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return node_input

  agent = Workflow(
      name='test_content_to_str_warns',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content(
      types.Content(
          parts=[
              types.Part(text='Hello'),
              types.Part(
                  inline_data=types.Blob(data=b'img', mime_type='image/png')
              ),
          ],
          role='user',
      )
  )
  with caplog.at_level(logging.WARNING):
    await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'Hello'
  assert 'non-text parts' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'produce_value, expected',
    [
        (lambda: [1, 2, 3], [1, 2, 3]),
        (lambda: {'key': 'value'}, {'key': 'value'}),
    ],
    ids=['list', 'dict'],
)
async def test_union_type_accepts_matching_member(
    request: pytest.FixtureRequest,
    produce_value,
    expected,
):
  """Tests that Union type annotation accepts values matching any member."""
  received_inputs = []

  def record_input(node_input: Union[list, dict]) -> str:
    received_inputs.append(node_input)
    return 'ok'

  agent = Workflow(
      name='test_union_accept',
      edges=[
          (START, produce_value),
          (produce_value, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('go'))

  assert len(received_inputs) == 1
  assert received_inputs[0] == expected


@pytest.mark.asyncio
async def test_optional_str_with_content_auto_conversion(
    request: pytest.FixtureRequest,
):
  """Tests that Optional[str] auto-converts Content from START node."""
  received_inputs = []

  def record_input(node_input: Optional[str]) -> str:
    received_inputs.append(node_input)
    return 'ok'

  agent = Workflow(
      name='test_optional_content',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('hello'))

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'hello'


@pytest.mark.asyncio
async def test_union_type_rejects_non_matching(
    request: pytest.FixtureRequest,
):
  """Tests that Union type annotation rejects values not matching any member."""

  def bad_input(node_input: Union[str, int]) -> str:
    return 'should not reach'

  def produce_list() -> list:
    return [1, 2, 3]

  agent = Workflow(
      name='test_union_reject',
      edges=[
          (START, produce_list),
          (produce_list, bad_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(TypeError, match='expects type'):
    await runner.run_async(testing_utils.get_user_content('go'))


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_sync(
    request: pytest.FixtureRequest,
):
  """Tests that state set via ctx.state in a sync function is persisted."""

  def set_state_via_ctx(ctx: Context) -> str:
    ctx.state['user_request'] = 'build a tracker app'
    return 'done'

  def read_state(user_request: str) -> str:
    return f'request={user_request}'

  agent = Workflow(
      name='test_ctx_state_delta_sync',
      edges=[
          (START, set_state_via_ctx),
          (set_state_via_ctx, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_sync',
          {
              'node_name': 'set_state_via_ctx',
              'output': 'done',
              'state_delta': {'user_request': 'build a tracker app'},
          },
      ),
      (
          'test_ctx_state_delta_sync',
          {
              'node_name': 'read_state',
              'output': 'request=build a tracker app',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_async(
    request: pytest.FixtureRequest,
):
  """Tests that state set via ctx.state in an async function is persisted."""

  async def set_state_via_ctx(ctx: Context) -> str:
    ctx.state['counter'] = 42
    ctx.state['name'] = 'test'
    return 'set'

  def read_state(counter: int, name: str) -> str:
    return f'counter={counter}, name={name}'

  agent = Workflow(
      name='test_ctx_state_delta_async',
      edges=[
          (START, set_state_via_ctx),
          (set_state_via_ctx, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_async',
          {
              'node_name': 'set_state_via_ctx',
              'output': 'set',
              'state_delta': {'counter': 42, 'name': 'test'},
          },
      ),
      (
          'test_ctx_state_delta_async',
          {
              'node_name': 'read_state',
              'output': 'counter=42, name=test',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_none_return(
    request: pytest.FixtureRequest,
):
  """Tests that state is persisted even when function returns None."""

  def set_state_return_none(ctx: Context) -> None:
    ctx.state['my_key'] = 'my_value'

  def read_state(my_key: str) -> str:
    return f'my_key={my_key}'

  agent = Workflow(
      name='test_ctx_state_delta_none_return',
      edges=[
          (START, set_state_return_none),
          (set_state_return_none, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_none_return',
          {
              'node_name': 'set_state_return_none',
              'state_delta': {'my_key': 'my_value'},
              'output': None,
          },
      ),
      (
          'test_ctx_state_delta_none_return',
          {
              'node_name': 'read_state',
              'output': 'my_key=my_value',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_with_event_return(
    request: pytest.FixtureRequest,
):
  """Tests that ctx.state changes merge into a returned Event's state_delta."""

  def set_state_return_event(ctx: Context) -> Event:
    ctx.state['from_ctx'] = 'ctx_value'
    return Event(
        data='result',
        state={'from_event': 'event_value'},
    )

  def read_state(from_ctx: str, from_event: str) -> str:
    return f'from_ctx={from_ctx}, from_event={from_event}'

  agent = Workflow(
      name='test_ctx_state_delta_event_return',
      edges=[
          (START, set_state_return_event),
          (set_state_return_event, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_event_return',
          {
              'node_name': 'set_state_return_event',
              'output': 'result',
              'state_delta': {
                  'from_event': 'event_value',
                  'from_ctx': 'ctx_value',
              },
          },
      ),
      (
          'test_ctx_state_delta_event_return',
          {
              'node_name': 'read_state',
              'output': 'from_ctx=ctx_value, from_event=event_value',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_generator(
    request: pytest.FixtureRequest,
):
  """Tests that ctx.state changes are captured in generator yields."""

  def gen_with_state(ctx: Context) -> Generator[Any, None, None]:
    ctx.state['key1'] = 'value1'
    yield 'first'
    ctx.state['key2'] = 'value2'
    yield 'second'

  def read_state(key1: str, key2: str) -> str:
    return f'key1={key1}, key2={key2}'

  agent = Workflow(
      name='test_ctx_state_delta_generator',
      edges=[
          (START, gen_with_state),
          (gen_with_state, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)

  # Both yields should have state_delta. The second yield includes
  # accumulated state from both ctx.state assignments.
  assert simplified == [
      (
          'test_ctx_state_delta_generator',
          {
              'node_name': 'gen_with_state',
              'output': 'first',
              'state_delta': {'key1': 'value1'},
          },
      ),
      (
          'test_ctx_state_delta_generator',
          {
              'node_name': 'gen_with_state',
              'output': 'second',
              'state_delta': {'key1': 'value1', 'key2': 'value2'},
          },
      ),
      (
          'test_ctx_state_delta_generator',
          {
              'node_name': 'read_state',
              'output': 'key1=value1, key2=value2',
          },
      ),
  ]
