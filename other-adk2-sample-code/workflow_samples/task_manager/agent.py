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

"""Task manager: RequestInput with response_schema and nested workflow.

Demonstrates: response_schema on RequestInput, rerun_on_resume,
ctx.resume_inputs, nested Workflow, conditional routing after HITL.

Usage::

    adk web contributing/workflow_samples/

Select "task_manager" in the web UI, then try:

    Sample queries:
      - "Build a mobile app for tracking daily water intake"
      - "Organize a company hackathon for 50 people"
      - "Migrate our monolith to microservices"

    When the approval form appears, respond with JSON:
      Approve:  {"approved": true, "feedback": ""}
      Reject:   {"approved": false, "feedback": "Add notifications task"}
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Edge
from google.adk.workflow import Workflow
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow.node import node
from pydantic import BaseModel
from pydantic import Field


class TaskPlan(BaseModel):
  project_name: str
  tasks: list[dict]


class TaskApproval(BaseModel):
  """Schema for the HITL approval response."""

  approved: bool = Field(description='Whether the plan is approved')
  feedback: str = Field(default='', description='Optional feedback')


def save_request(node_input: str):
  """Store user request in state for replan loop."""
  yield Event(state={'user_request': node_input})


plan_agent = LlmAgent(
    name='plan_tasks',
    model='gemini-2.5-flash',
    instruction=(
        'Break down this project into 3-5 tasks with title, description,'
        ' priority (high/medium/low), and estimated_hours.'
    ),
    output_key='task_plan',
    output_schema=TaskPlan,
)


def _unwrap_response(response: dict) -> dict:
  """Unwrap web UI response wrapper {"response": ...} if present."""
  data = response.get('response', response)
  if isinstance(data, str):
    data = json.loads(data)
  return data


async def review_tasks(ctx: Context, node_input: Any):
  """HITL: ask user to approve the task plan via response_schema."""
  # Use a unique interrupt_id per review cycle so that event-based state
  # reconstruction can distinguish responses across rejection loops.
  review_count = ctx.state.get('review_count', 0)
  interrupt_id = f'review_{review_count}'

  response = ctx.resume_inputs.get(interrupt_id)
  if response:
    data = _unwrap_response(response)
    feedback = data.get('feedback', '')
    route = 'approved' if data.get('approved') else 'rejected'
    yield Event(
        data=data,
        route=route,
        state={'feedback': feedback, 'review_count': review_count + 1},
    )
    return

  plan = ctx.state.get('task_plan', {})
  tasks = plan.get('tasks', [])
  lines = [f'Project: {plan.get("project_name", "?")}', '']
  for i, t in enumerate(tasks, 1):
    lines.append(f'{i}. [{t.get("priority")}] {t.get("title")}')
  lines.append('\nApprove this plan?')

  yield RequestInput(
      interrupt_id=interrupt_id,
      message='\n'.join(lines),
      response_schema=TaskApproval.model_json_schema(),
  )


review_node = node(review_tasks, rerun_on_resume=True)


def prepare_replan(ctx: Context, node_input: Any) -> str:
  """Format replan prompt with original request and rejection feedback."""
  request = ctx.state.get('user_request', '')
  feedback = ctx.state.get('feedback', '')
  return f'Project: {request}\nRevise based on feedback: {feedback}'


# --- Nested workflow: execute tasks ---


def prepare_summary(ctx: Context, node_input: Any) -> str:
  """Build context string for the summarize LLM from state."""
  plan = ctx.state.get('task_plan', {})
  tasks = plan.get('tasks', [])
  assignments = [f'- {t.get("title", "?")} -> Team' for t in tasks]
  feedback = ctx.state.get('feedback', '')
  parts = [
      f'Project: {plan.get("project_name", "?")}',
      f'Tasks assigned:\n' + '\n'.join(assignments),
  ]
  if feedback:
    parts.append(f'User feedback: {feedback}')
  return '\n\n'.join(parts)


summarize_agent = LlmAgent(
    name='summarize',
    model='gemini-2.5-flash',
    instruction='Summarize the task execution plan and assignments below.',
    output_key='summary',
)

execute_pipeline = Workflow(
    name='execute_pipeline',
    edges=[('START', prepare_summary), (prepare_summary, summarize_agent)],
)

# --- Main workflow ---

root_agent = Workflow(
    name='task_manager',
    edges=[
        ('START', save_request),
        (save_request, plan_agent),
        (plan_agent, review_node),
        (review_node, {'approved': execute_pipeline, 'rejected': prepare_replan}),
        (prepare_replan, plan_agent),
    ],
)

app = App(
    name='task_manager',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
