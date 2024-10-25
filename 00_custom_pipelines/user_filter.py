from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
 
 
class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
 
    def __init__(self):
        self.type = "filter"
        self.name = "User Info Pipeline"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )
 
    async def on_startup(self):
        print(f"on_startup: {__name__}")
 
    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
 
    async def on_valves_updated(self):
        pass
 
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet: {__name__}")
        messages = body["messages"]
        # Modify the first message to include user info if not already done
        if len(messages) > 0:
            first_message = messages[0]
            if first_message["role"] == "user" and "Sent by:" not in first_message["content"]:
                # Get the user's name
                user_name = user.get("name", "User") if user else "User"
 
                # Get current local date and time
                current_datetime = datetime.now().strftime("%Y-%m-%d")
                # Append user name and timestamp to the first message
                first_message["content"] = (
                    f"{first_message['content']}\n\nSent by: {user_name} at {current_datetime}"
                )
                messages[0] = first_message
 
        body["messages"] = messages
        return body