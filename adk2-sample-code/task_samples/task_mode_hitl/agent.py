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

"""Task mode HITL: flight booking with ADK human-in-the-loop tools.

Demonstrates task delegation combined with ADK HITL mechanisms:
  - get_user_choice_tool (LongRunningFunctionTool): pauses execution
    and waits for user to pick a flight from presented options.
  - tool_context.request_confirmation(): asks user to approve/reject
    before executing (e.g. confirming a booking).

Flow:
  1. Coordinator delegates a flight booking task.
  2. Flight agent searches for flights (regular tool).
  3. Flight agent calls get_user_choice(options=[...]) which pauses
     and waits for user to pick a flight.
  4. User provides their choice as the function response.
  5. Flight agent calls book_flight which requires confirmation
     (request_confirmation) before executing.
  6. User confirms the booking.
  7. Flight agent calls finish_task with confirmation details.

Run with:
  adk web contributing/task_samples/
"""

from pydantic import BaseModel

from google.adk.tools.get_user_choice_tool import get_user_choice_tool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow.agents.llm_agent import Agent

# -- Schemas ----------------------------------------------------------------


class FlightBookingInput(BaseModel):
  """Input schema for the flight booking agent."""

  origin: str
  destination: str
  date: str


class FlightBookingOutput(BaseModel):
  """Output schema for the flight booking agent."""

  confirmation_code: str
  airline: str
  flight_number: str
  passenger_name: str
  price: str


# -- Tools ------------------------------------------------------------------


def search_flights(origin: str, destination: str, date: str) -> str:
  """Search for available flights.

  Args:
    origin: Departure city or airport code.
    destination: Arrival city or airport code.
    date: Travel date (YYYY-MM-DD).

  Returns:
    Available flight options.
  """
  return (
      f'Flights from {origin} to {destination} on {date}:\n'
      '1. [UA801] United - $850 - Departs 10:30 AM, Arrives 3:15 PM+1\n'
      '2. [JL002] Japan Airlines - $920 - Departs 1:00 PM,'
      ' Arrives 5:30 PM+1\n'
      '3. [AA175] American - $780 - Departs 6:00 PM,'
      ' Arrives 10:45 PM+1 (1 stop)'
  )


def book_flight(
    flight_id: str,
    passenger_name: str,
    tool_context: ToolContext,
) -> str:
  """Book a flight for a passenger. Requires user confirmation.

  Args:
    flight_id: The flight identifier to book (e.g. UA801).
    passenger_name: Full name of the passenger.
    tool_context: The tool context (injected by framework).

  Returns:
    Booking confirmation details.
  """
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(
        hint=f'Confirm booking flight {flight_id} for {passenger_name}?',
    )
    tool_context.actions.skip_summarization = True
    return 'Awaiting booking confirmation.'

  if not tool_context.tool_confirmation.confirmed:
    return 'Booking cancelled by user.'

  code = f'TRV-{abs(hash(flight_id + passenger_name)) % 100000:05d}'
  return (
      'Booking confirmed!\n'
      f'Confirmation: {code}\n'
      f'Flight: {flight_id}\n'
      f'Passenger: {passenger_name}\n'
      'Status: Confirmed - e-ticket will be sent to your email.'
  )


# -- Agents ----------------------------------------------------------------

flight_booker = Agent(
    name='flight_booker',
    mode='task',
    input_schema=FlightBookingInput,
    output_schema=FlightBookingOutput,
    instruction=(
        'You are a flight booking assistant. When given a booking'
        ' request:\n'
        '1. Use search_flights to find available options.\n'
        '2. Use get_user_choice to present the flight options and'
        ' wait for the user to pick one. Pass the options as a list'
        ' of strings like ["UA801 - United $850", "JL002 - JAL $920",'
        ' "AA175 - American $780"].\n'
        '3. Once the user picks a flight, use book_flight with the'
        " flight_id and the user's full name. The book_flight tool"
        ' will ask for user confirmation before proceeding.\n'
        '4. After the booking is confirmed, call finish_task with the'
        ' confirmation details.\n\n'
        'Do NOT skip the get_user_choice step. Always let the user'
        ' choose before booking.'
    ),
    description=(
        'Books flights interactively using HITL tools. Pauses for'
        ' user selection and requires booking confirmation.'
    ),
    tools=[search_flights, get_user_choice_tool, book_flight],
)

root_agent = Agent(
    name='travel_coordinator',
    model='gemini-2.5-flash',
    sub_agents=[flight_booker],
    instruction=(
        'You are a travel coordinator. When the user wants to book a'
        ' flight, delegate to the flight_booker agent using'
        ' request_task_flight_booker. The flight booker will interact'
        ' directly with the user to search flights, present options,'
        ' and confirm the booking.\n\n'
        'After the booking is complete, present the confirmation'
        ' details to the user.'
    ),
)
