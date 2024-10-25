from typing import List, Optional, Callable, Any, Awaitable
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime
import os
import uuid

from utils.pipelines.main import get_last_user_message, get_last_assistant_message


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 10
        mongo_uri: str
        
    def __init__(self):
        self.type = "filter"
        self.name = "MongoDB Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "mongo_uri": "mongodb://admin:hey_c0r0n4@mongodb:27017/",
            }
        )
        self.client = None
        self.db = None
        self.collection = None
        
    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_mongo_client()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        if self.client:
            self.client.close()
            print(f"{self.name} MongoDB connection closed.")
        else:
            print(f"{self.name} No active MongoDB connection to close.")

    async def on_valves_updated(self):
        self.set_mongo_client()

    def set_mongo_client(self):
        try:
            self.client = MongoClient(self.valves.mongo_uri)
            self.db = self.client['hey_corona']
            self.collection = self.db['chat_logs']
            
            # Create indexes on chat_id and user_name
            self.collection.create_index("message_id", unique=True)
            self.collection.create_index("chat_id")
            self.collection.create_index("user_id")
            self.collection.create_index("user_name")
            self.collection.create_index("model")


        except Exception as e:
            print(f"{self.name} MongoDB connection error: {e}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        try:
            if "chat_id" not in body:
                unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
                body["chat_id"] = unique_id
                print(f"chat_id was missing, set to: {unique_id}")

            # Create a new chat log document
            if 'id' in body:
                chat_log = {
                    "message_id": body['id'],
                    "chat_id": body["chat_id"],
                    "user_id": user["id"] if user else "unknown",
                    "user_name": user["name"] if user else "unknown",
                    "model": body["model"],
                    "messages": body["messages"],
                    "last_updated": datetime.now()
                }
                self.collection.insert_one(chat_log)
            return body
        
        except Exception as e:
            print(f"{self.name} Inlet error: {e}")
            print(body)
            print(user)
            return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        try:
            if "chat_id" not in body:
                print("Error in outlet: Missing chat_id")
                return body
            # print('body', body)
            # print('user', user)
            user_message = get_last_user_message(body["messages"])
            generated_message = get_last_assistant_message(body["messages"])

            print_dict = {'user_message':user_message, 'generated_message':generated_message}
            print_dict['user'] = user["name"] if "name" in user else 'unknown'

            # print(f'PROMPT:   {print_dict}')
            
            update = {
                "messages": body["messages"],  # Replace the entire messages array
                "last_user_message": user_message,
                "last_generated_message": generated_message,
                "last_updated": datetime.now()
            }

            self.collection.update_one(
                {"message_id": body["id"]},
                {"$set": update}
            )

            return body
        except Exception as e:
            print(f"{self.name} Outlet error: {e}")
            print(body)
            print(user)

            return body

