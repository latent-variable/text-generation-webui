from typing import List, Optional
from pydantic import BaseModel
import os
import requests
import json

from utils.pipelines.main import get_last_user_message
 

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for function calling
        OLLAMA_BASE_URL: str



    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "function_calling_blueprint"
        self.name = "Function Calling Blueprint"

        # Initialize valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            }
        )

        self.pipelines = [
            {
                "id": "vanna_txt_to_sql_pipeline",  # This will turn into manifold_pipeline.pipeline-1
                "name": "Postgres Vanna SQL Pipeline",  # This will turn into Manifold: Pipeline 1
            }, 
            {
                "id": "vanna_txt_to_sql_pipeline",  # This will turn into manifold_pipeline.pipeline-1
                "name": "Postgres Vanna SQL Pipeline",  # This will turn into Manifold: Pipeline 1
            }
        ]

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"INLET:{__name__}")
        # If title generation is requested, skip the function calling filter
        if body.get("title", False):
            return 
        
        if 'metadata' in body:
            metadata = body['metadata']
            print('fasdf')
            if 'task' in metadata:
                print('fasdaddf')
                if 'title_generation' == metadata['task']:
                    print('fasdaddadf')
                    body["title"] = True
                    return body
        
        print('*************BODY:', body)

        # Get the last user message
        user_message = get_last_user_message(body["messages"])

        # Get the tools specs
        # tools_specs = get_tools_specs(self.tools)
        pipeline_specs = self.pipelines
        
        # System prompt for function calling
        fc_system_prompt = (
            f"Pipelines: {json.dumps(pipeline_specs, indent=2)}" + 
            """
            You are given a list of pipelines and models. Your task is to choose the best pipeline or model ID based on the user query. Here are the instructions:

            - If the query is best matched by one of the pipelines listed return the corresponding pipeline ID.

            - If a query seems to relate to naval engagements, data relating to the navy or federal body, analystic requests, and an SQL query can be made to a database if the query were to be transformed into an SQL statement, it is most likely the vanna_txt_to_sql_pipeline.

            - If the query is general or doesn't match any of the listed pipelines, return the ID of a conversational model, such as phi3:mini or llama3.1:latest, whichever is more suit for the prompt

            - Always return the model ID as a string. DO NOT INCLUDE ANYTHING ELSE. THIS MEANS NO EXPLANATION OR ANY ADDITIONAL TEXT. Just return the model ID

            Example Queries:
            1. "List me all ships within the database." → Return: vanna_txt_to_sql_pipeline
            4. "What are the coordinates off the USS Arleigh Burke?" → Return: vanna_txt_to_sql_pipeline

            If you are unsure which pipeline to use, default to returning a conversational model.
            Again, Never return anything more than the model ID. Only generate ID's that have been given to you. 
            """
        )

        r = None
        task_model = body['model']
        # Extract the last 4 messages in reverse order
        for message in body["messages"][::-1][:4]:
            print(f"{message['role']}: {message['content']}")
        try:
            # Call the OpenAI API to get the function response
            r = requests.post(
                url=f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": task_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": fc_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": "History:\n"
                            + "\n".join(
                                [
                                    f"{message['role']}: {message['content']}"
                                    for message in body["messages"][::-1][:4]
                                ]
                            )
                            + f"Query: {user_message}",
                        },
                    ]
                    # : dynamically add response_format?
                    # "response_format": {"type": "json_object"},
                },
                stream=False,
            )
            r.raise_for_status()

            response = r.json()
            content = response["choices"][0]["message"]["content"]
            t = response["choices"][0]

            print(f"REPONSE:\n{response}\n--------------------------")
            model = body["model"]
            print(f"OLD MODEL USED:\n{model}\n--------------------------")
            body["model"] = content
            print(f"FILTIRED PIPELINE USED:\n{content}\n--------------------------")
            t = body
            print(f"BODY\n: {t}\n--------------------------")
            
            return {**body, "model": content}

        except Exception as e:
            print(f"Error: {e}")
        return body