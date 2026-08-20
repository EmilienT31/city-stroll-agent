# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.city_stroll import generate_paths


MODEL = "gemini-3.6-flash"


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are City Stroll Agent. Create seamless, taste-matched walking
itineraries by calling generate_paths. Collect a city and the traveler's temporary
tastes before calling the tool. Never invent venues, Place IDs, distances, opening
hours, or Maps links; only present fields returned by the tool. Compare the three
alternatives concisely, preserve all caveats, and do not book, purchase, reserve, or
start navigation. Treat allergy, accessibility, dietary, budget, and opening-hour
details as preferences that the traveler must verify directly.""",
    tools=[generate_paths],
)

app = App(
    root_agent=root_agent,
    name="app",
)
