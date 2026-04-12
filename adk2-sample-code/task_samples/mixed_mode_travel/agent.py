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

"""Mixed-mode travel planner: task + single-turn children.

Demonstrates mixed delegation patterns with a travel planner:
  - flight_searcher (mode='task'): interactive -- the user can discuss
    flight options, change preferences, and confirm before booking.
  - weather_checker (mode='single_turn'): autonomous -- checks weather
    for the destination with no user interaction.
  - hotel_finder (mode='single_turn'): autonomous -- finds hotel
    options with no user interaction.

The coordinator can delegate to weather_checker and hotel_finder
autonomously, while flight_searcher allows interactive discussion.

Run with:
  adk web contributing/task_samples/
"""

from google.adk.workflow.agents.llm_agent import Agent
from pydantic import BaseModel

# -- Schemas ----------------------------------------------------------------


class FlightSearchInput(BaseModel):
  """Input for the flight searcher."""

  origin: str
  destination: str
  departure_date: str
  return_date: str = ''


class FlightSearchOutput(BaseModel):
  """Output from the flight searcher."""

  airline: str
  price: str
  departure_time: str
  arrival_time: str
  confirmed: bool


class WeatherOutput(BaseModel):
  """Output from the weather checker."""

  temperature: str
  conditions: str
  recommendation: str


class HotelOutput(BaseModel):
  """Output from the hotel finder."""

  hotel_name: str
  price_per_night: str
  rating: str
  location: str


# -- Tools ------------------------------------------------------------------


def search_flights(
    origin: str,
    destination: str,
    date: str,
) -> str:
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
      '1. United UA123 - $450 - Departs 8:00 AM, Arrives 11:30 AM\n'
      '2. Delta DL456 - $380 - Departs 2:00 PM, Arrives 5:45 PM\n'
      '3. American AA789 - $520 - Departs 6:00 AM, Arrives 9:15 AM'
  )


def book_flight(flight_number: str, passenger_name: str) -> str:
  """Book a specific flight.

  Args:
    flight_number: The flight number to book (e.g. UA123).
    passenger_name: Name of the passenger.

  Returns:
    Booking confirmation.
  """
  return (
      f'Booking confirmed for {passenger_name} on flight'
      f' {flight_number}. Confirmation: TRV-{hash(flight_number) % 10000:04d}'
  )


def get_weather(city: str) -> str:
  """Get current weather for a city.

  Args:
    city: The city name.

  Returns:
    Weather information.
  """
  return (
      f'Weather in {city}:\n'
      'Temperature: 72F (22C)\n'
      'Conditions: Partly cloudy\n'
      'Humidity: 55%\n'
      'Forecast: Pleasant weather expected for the next 5 days.'
  )


def find_hotels(city: str, check_in: str) -> str:
  """Find hotels in a city.

  Args:
    city: The city to search in.
    check_in: Check-in date (YYYY-MM-DD).

  Returns:
    Available hotel options.
  """
  return (
      f'Hotels in {city} (check-in {check_in}):\n'
      '1. Grand Hotel - $180/night - 4.5 stars - Downtown\n'
      '2. Comfort Inn - $95/night - 3.8 stars - Airport area\n'
      '3. Luxury Suites - $320/night - 4.9 stars - Waterfront'
  )


# -- Agents ----------------------------------------------------------------

flight_searcher = Agent(
    name='flight_searcher',
    mode='task',
    input_schema=FlightSearchInput,
    output_schema=FlightSearchOutput,
    instruction=(
        'You are a flight search specialist. When given flight'
        ' search criteria:\n'
        '1. Use search_flights to find options.\n'
        '2. Present the options to the user and discuss preferences.\n'
        '3. If the user wants to book, use book_flight.\n'
        '4. When the user confirms or you have found the best option,'
        ' call finish_task with the flight details.\n'
        'You can chat with the user to clarify preferences like'
        ' preferred airlines, times, or budget.'
    ),
    description=(
        'Searches and books flights interactively. Discusses options'
        ' with the user before confirming.'
    ),
    tools=[search_flights, book_flight],
)

weather_checker = Agent(
    name='weather_checker',
    mode='single_turn',
    output_schema=WeatherOutput,
    instruction=(
        'You check the weather for a destination city.\n'
        '1. Use get_weather to look up the conditions.\n'
        '2. Call finish_task with temperature, conditions, and a'
        ' packing recommendation.\n'
        'Complete this autonomously without user interaction.'
    ),
    description='Checks weather conditions for a destination.',
    tools=[get_weather],
)

hotel_finder = Agent(
    name='hotel_finder',
    mode='single_turn',
    output_schema=HotelOutput,
    instruction=(
        'You find hotel options for a destination.\n'
        '1. Use find_hotels to search for options.\n'
        '2. Pick the best value option and call finish_task with'
        ' hotel_name, price_per_night, rating, and location.\n'
        'Complete this autonomously without user interaction.'
    ),
    description='Finds and recommends hotels for a destination.',
    tools=[find_hotels],
)

root_agent = Agent(
    name='travel_planner',
    model='gemini-2.5-flash',
    sub_agents=[flight_searcher, weather_checker, hotel_finder],
    instruction=(
        'You are a travel planning coordinator. Help users plan trips'
        ' by delegating to your specialist agents:\n'
        '- Use request_task_weather_checker to check destination'
        ' weather (autonomous, no user interaction needed).\n'
        '- Use request_task_hotel_finder to find hotel options'
        ' (autonomous, no user interaction needed).\n'
        '- Use request_task_flight_searcher for flight search and'
        ' booking (interactive, the user can discuss options).\n\n'
        'For a full trip plan, start by checking weather and finding'
        ' hotels, then help the user search for flights. Present all'
        ' results together in a comprehensive travel summary.'
    ),
)
