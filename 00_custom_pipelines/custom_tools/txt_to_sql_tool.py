from typing import List, Optional, Union, Generator, Iterator, Callable, Awaitable
from pydantic import BaseModel, Field
from tabulate import tabulate
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import sqlparse
import json


def generate_sql_dropdown(formatted_sql):
    markdown_dropdown = f"```sql\n{formatted_sql};\n```"
    return markdown_dropdown


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


class HelperFunctions:
    def __init__(self):
        self.vn = None
        self.is_trained = False  # Flag to check if training has occurred

    def connect_vanna_pg(self, model):
        if self.vn is not None:
            print("Vanna is already connected.")
            return
        try:
            print("Step 1: Instantiating self.vn with a MyVannaa Object...")
            self.vn = MyVanna(
                config={
                    "model": model,
                    "ollama_host": "http://host.docker.internal:11434",
                    "allow_llm_to_see_data": True,
                }
            )
            print("Step 1: Success\nStep 2: Connecting to postgres DB")

            self.vn.connect_to_postgres(
                host="postgres-metrics",
                dbname="metricdb",
                user="metrics",
                password="metrics",
                port=5432,
            )
            print("Vanna connected to postgres-metrics")
        except Exception as e:
            print(f"Failed to connect to postgres DB: {e}")

    def open_json_file(self, FILE_INPUT):
        try:
            print("Opening file")
            with open(FILE_INPUT, "r") as file:
                data = json.load(file)
                print("File successfully opened")
                return data
        except FileNotFoundError:
            raise Exception(f"File not found: {FILE_INPUT}")
        except PermissionError:
            raise Exception(f"Permission denied for file: {FILE_INPUT}")
        except Exception as e:
            raise Exception(f"Error opening file: {e}")

    def clear_embeddings(self):
        try:
            training_data = self.vn.get_training_data()
            if training_data is not None and "id" in training_data:
                for row_id in training_data["id"]:
                    self.vn.remove_training_data(id=row_id)
                print("Successfully removed embeddings")
            else:
                print("No embeddings to remove.")
        except Exception as e:
            print(f"Failed to clear embeddings: {e}")

    def execute_training(self, FILE_INPUT):
        try:
            data = self.open_json_file(FILE_INPUT)
            print("Attempting to train data...")
            training_data = self.vn.get_training_data()

            if training_data is not None:
                self.clear_embeddings()

            df_information_schema = self.vn.run_sql(
                "SELECT * FROM INFORMATION_SCHEMA.COLUMNS"
            )
            plan = self.vn.get_training_plan_generic(df_information_schema)

            self.vn.train(
                ddl=data["ddl"], documentation=data["documentation"], plan=plan
            )

            # Loop through question_sql_pairs and add them
            for pair in data.get("question_sql_pairs", []):
                print(f"Adding question: {pair['question']}")
                self.vn.add_question_sql(question=pair["question"], sql=pair["sql"])
            print("Training completed successfully.")
            self.check_if_trained = True
            self.is_trained = True
        except Exception as e:
            print(f"Failed to train data: {e}")

    def check_if_trained(self):
        try:
            # Check if embeddings exist
            training_data = self.vn.get_training_data()
            if training_data is not None and len(training_data.get("id", [])) > 0:
                print("Data already trained...")
                self.is_trained = True
                return True
            else:
                return False
        except Exception as e:
            print(f"Failed to check training status: {e}")
            return False


class Tools:
    class Valves(BaseModel):
        NAME: str = Field(
            default="Postgres Vanna SQL Pipeline",
            description="The name of this SQL pipeline.",
        )
        FILE_INPUT: str = Field(
            default="/media/training.json",
            description="The path to the JSON file containing training data.",
        )
        MODEL: str = Field(
            default="text2sql:latest",
            description="The LLM model to be used for generating SQL queries.",
        )
        OLLAMA_BASE_URL: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for the Ollama API.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def perform_txt_to_sql(
        self, prompt: str, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        Generate a targeted SQL statement given a prompt requesting specific insights from naval data.

        This function is optimized to interpret natural language prompts that strictly query insights, metrics, 
        statistics, or other analytical data without modifying the database.

        **Warning:** This function is restricted to SELECT statements only. It will not process requests that 
        attempt to insert, update, delete, or otherwise alter the database. Requests should focus on 
        retrieving data such as ship performance, location tracking, or inventory management insights.

        :param prompt: A request in natural language specifically seeking insights, metrics, statistics, 
                    or analytical data from the naval database, such as ship performance, location tracking, 
                    or inventory management. Modifying actions (e.g., INSERT, UPDATE, DELETE) are not permitted.
        """
        try:
            functions = HelperFunctions()
            functions.connect_vanna_pg(self.valves.MODEL)
            if functions.check_if_trained == False:
                functions.execute_training(self.valves.FILE_INPUT)
            else:
                functions.clear_embeddings()
                functions.execute_training(self.valves.FILE_INPUT)

            if functions.vn is None:
                print("Error: Vanna is not connected.")
                return "Error: Vanna is not connected."
            # Generate the SQL, execute it, and create a plot
            sql, result, plot = functions.vn.ask(
                question=prompt, allow_llm_to_see_data=True
            )
            print("sql TYPE:", type(sql))

            # Format the output string
            formatted_sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            sql_query = generate_sql_dropdown(formatted_sql)
            markdown_table = tabulate(result, headers="keys", tablefmt="pipe")
            # response = sql_query + "\n\n" + markdown_table + "\n\n"
            response = markdown_table + "\n\n"


            # if plot:
            #     # Convert the Plotly figure to a PNG image
            #     img_bytes = plot.to_image(format="png")
            #     # Encode the image in Base64
            #     import base64
            #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            #     # Embed the image in the Markdown response
            #     response += f'![Visualization](data:image/png;base64,{img_base64})\n'

            await __event_emitter__(
                {
                    "type": "message",  # We set the type here
                    # "data": {"content": "Generating SQL Statement...\n"},
                    "data": {"content": sql_query + "\n"},
                    # Note that with message types we do NOT have to set a done condition
                }
            )
            print(f"Prompt:\n{prompt}")
            return response

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"
