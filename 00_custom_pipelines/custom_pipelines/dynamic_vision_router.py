"""
title: Dynamic Vision Router

"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json


def get_last_message(messages):
    if messages:  # Check if the messages list is not empty
        return messages[-1]  # Return the last message dictionary
    return None  # Return None if the list is empty

class Pipeline:
    class Valves(BaseModel):
        priority: int = 0
        vision_model_id: str = Field(
            default="llava-phi3:latest",
            description="The identifier of the vision model to be used for processing images. Note: Compatibility is provider-specific; ollama models can only route to ollama models, and OpenAI models to OpenAI models respectively.",
        )
        status: bool = Field(
            default=False,
            description="A flag to enable or disable the status indicator. Set to True to enable status updates.",
        )
        pipelines: list = ["*"]
        pass

    def __init__(self):
        self.type = "filter"
        self.name = "Vision Router"
        self.valves = self.Valves()
        pass
    
    async def on_startup(self):
        print(f"on_startup:{__name__}")


    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
      

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"INLET:{__name__}")
   
        if body['model'] == self.valves.vision_model_id:
            return body

        messages = body["messages"]
        user_message = get_last_message(messages)

        if user_message is None:
            return body

        has_images = "images" in user_message or (
            isinstance(user_message.get("content"), list)
            and any(item.get("type") == "image_url" for item in user_message["content"])
        )
        if has_images:
            if self.valves.vision_model_id:
                body["model"] = self.valves.vision_model_id
           
        return body
