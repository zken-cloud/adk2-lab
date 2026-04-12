# Mixed-Mode Travel Planner

Demonstrates mixed delegation patterns with a travel planner coordinator
and three specialist children:

- **flight_searcher** (`mode='task'`): Interactive -- searches and books
  flights. The user can discuss options and preferences before confirming.
- **weather_checker** (`mode='single_turn'`): Autonomous -- checks
  weather for the destination with no user interaction.
- **hotel_finder** (`mode='single_turn'`): Autonomous -- finds hotel
  options with no user interaction.

## Run

```bash
adk web contributing/task_samples/
```

Select **mixed_mode_travel** in the web UI.

## Testing Prompts

- Plan a trip to Tokyo next week
- Find flights from SFO to London and check the weather
- I need a hotel in Paris and a flight from NYC
- What's the weather like in Barcelona? Also find me a hotel there

- Book flights for me and my wife from SFO to Tokyo, 2026-07-10 --> 2026-08-10
