from pydantic import BaseModel, Field
from typing import Optional
import re
import psycopg2
from tabulate import tabulate

class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def run_sql(self, sql):
        conn = psycopg2.connect(
            dbname="metricdb",
            user="metrics",
            password="metrics",
            host="postgres-metrics",
            port="5432"
        )

        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            markdown_table = tabulate(result, headers=column_names, tablefmt='pipe')
        conn.close()
        return markdown_table
    
    def detect_table(self, last_message):
        table_pattern = r"(\|.*)"
        matches = re.findall(table_pattern, last_message, re.DOTALL)
        return len(matches) > 0
    
    def detect_sql(self, last_message):
        pattern = r"```sql\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, last_message)
        if len(matches) > 0:
            sql = matches[0].strip()
            return sql
        else:
            return None
        
    def detect_valid_sql(self, sql):
        modification_keywords = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE)\b"
        valid_sql = not re.search(modification_keywords, sql, re.IGNORECASE)
        return valid_sql

    async def action(self, body: dict, __user__=None, __event_emitter__=None, __event_call__=None,) -> Optional[dict]:
        print(f"action:{__name__}")
        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        last_assistant_message = body["messages"][-1]

        # if user_valves.show_status:
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Processing your input", "done": False},
            }
        )

        # Execute SQL query if the input is detected as code
        input_text = last_assistant_message["content"]
        print('input_text:\n', input_text, '\n------------------------')
        sql = self.detect_sql(input_text)
        
        if not sql:  # Early return if no SQL detected
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "No SQL query detected.",
                        "done": True,
                    },
                }
            )
            return

        sql_is_valid = self.detect_valid_sql(sql)
        table_exist = self.detect_table(input_text)

        if not sql_is_valid:  # Avoid double negation
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Illegal modify SQL queries detected.",
                        "done": True,
                    },
                }
            )
        elif not table_exist:
            output = '\n' + self.run_sql(sql)
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": output},
                }
            )
        else:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Table already generated",
                        "done": True,
                    },
                }
            )