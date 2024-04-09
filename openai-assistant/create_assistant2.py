import asyncio
import json
import os
import instructor
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
aclient = instructor.patch(AsyncOpenAI(api_key=api_key))


class WeatherParams(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    format: Literal["celsius", "fahrenheit"] = Field(
        description="The temperature unit to use. Infer this from the user's location."
    )


class ForecastParams(WeatherParams):
    num_days: int = Field(description="The number of days to forecast")


def get_current_weather(params: WeatherParams):
    return f"The current weather in {params.location} is 20 degrees {params.format}"


def get_n_day_weather_forecast(params: ForecastParams):
    return f"The weather forecast for the next {params.num_days} days in {params.location} is 20 degrees {params.format}"


async def create_assistant():
    assistant = await aclient.beta.assistants.create(
        name="Math Tutor And Weather Bot",
        instructions="""You are a personal math tutor. Write and run code to answer math questions.
Enclose math expressions in $$ (this is helpful to display latex). Example:
```
Given a formula below $$ s = ut + \frac{1}{2}at^{2} $$ Calculate the value of $s$ when $u = 10\frac{m}{s}$ and $a = 2\frac{m}{s^{2}}$ at $t = 1s$
```
You can also answer weather questions!
""",
        tools=[
            {"type": "code_interpreter"},
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": WeatherParams.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": ForecastParams.model_json_schema(),
                },
            },
        ],
        model="gpt-4-1106-preview",
    )
    return assistant


async def save_assistant(assistant):
    assistant_name = "math_tutor_and_weather_bot"

    def load_or_create_json(filename):
        try:
            return json.load(open(filename, "r"))
        except FileNotFoundError:
            return {}

    assistant_dict = load_or_create_json("assistants.json")
    assistant_dict[assistant_name] = assistant.id
    json.dump(assistant_dict, open("assistants.json", "w"))


async def main():
    assistant = await create_assistant()
    await save_assistant(assistant)


if __name__ == "__main__":
    asyncio.run(main())
    # print(json.dumps(WeatherParams.model_json_schema(), indent=2))
