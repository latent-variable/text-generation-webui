"""
title: task_router 
author: latent-variable
author_url: https://github.com/latent-variable
description:Route task to a specified model to be used for generating titles, tags, and autocompletes when one of those request are detected.
version: 0.0.1
"""

from pydantic import BaseModel, Field
from typing import Optional



def get_last_message(messages):
    if messages:  # Check if the messages list is not empty
        return messages[-1]  # Return the last message dictionary
    return None  # Return None if the list is empty


class Filter:
    class Valves(BaseModel):
        priority: int = 1
        title_model_id: str = Field(
            default="llama3.2:latest",
            description="The identifier of the model to be used for generating titles, tags, and autocompletes when one of those request are detected.",
        )
        status: bool = Field(
            default=True,
            description="A flag to enable or disable the status indicator. Set to True to enable status updates.",
        )
        pipelines: list = ["*"]
        pass

    def __init__(self):
        self.type = "filter"
        self.name = "Open-webui Task Router"
        self.valves = self.Valves()
        pass
    
    async def on_startup(self):
        print(f"on_startup:{__name__}")


    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
      

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"INLET:{__name__}")    
        if not self.valves.status:
            return body 
        

        messages = body["messages"]
        user_message = get_last_message(messages)

        if user_message is None:
            return body

        # Check if the prompt is a title request by matching with key phrases
        prompt = user_message.get("content", "")

        # If it's a title request, switch to the title model
        if "Create a concise" in prompt and "title" in prompt and 'Evolution of Music Streaming' in prompt:
            body["model"] = self.valves.title_model_id.strip()
            
        # If it's a autocompletion request, switch to the title model
        if "### Task:\nYou are an autocompletion system." in prompt:
            body["model"] = self.valves.title_model_id.strip()

        # If it's a tags request, switch to the title model
        if "### Task:\nGenerate 1-3 broad tags categorizing the main themes of the chat history," in prompt: 
            body["model"] = self.valves.title_model_id.strip()
       
        return body
