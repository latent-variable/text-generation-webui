from typing import List, Optional, Union, Generator, Iterator
from pydantic import BaseModel
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from tabulate import tabulate
from pathlib import Path
import json
import sqlparse
import requests
import os  # Added for checking persistent state



def generate_sql_dropdown(formatted_sql):
    markdown_dropdown = f"```sql\n{formatted_sql};\n```"
    return markdown_dropdown

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

class Pipeline:
    class Valves(BaseModel):
        NAME: str
        FILE_INPUT: Optional[Path]
        MODEL: str
        OLLAMA_BASE_URL: str
        pass

    def __init__(self):
        self.valves = self.Valves(
            **{
                "NAME": "Postgres Vanna SQL Pipeline",
                "FILE_INPUT": "/media/training.json",
                "MODEL": "hey_corona:latest",
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            }
        )
        self.name = self.valves.NAME
        self.vn = None
        self.is_trained = False  # Flag to check if training has occurred

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.connect_vanna_pg()
        if not self.check_if_trained():
            self.execute_training()
            self.is_trained = True  # Update the flag after training
        else:
            print("Model already trained. Skipping training on startup.")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.connect_vanna_pg()
        self.execute_training()
        self.is_trained = True  # Update the flag after training
        self.name = self.valves.NAME

    def connect_vanna_pg(self):
        if self.vn is not None:
            print("Vanna is already connected.")
            return
        try:
            self.vn = MyVanna(config={
                'model': 'hey_corona:latest',
                'ollama_host': 'http://host.docker.internal:11434',
                'allow_llm_to_see_data': True
            })

            self.vn.connect_to_postgres(
                host='postgres-metrics',
                dbname='metricdb',
                user='metrics',
                password='metrics',
                port=5432
            )
            print('Vanna connected to postgres-metrics')
        except Exception as e:
            print(f"Failed to connect to postgres DB: {e}")

    def open_json_file(self):
        try:
            print('Opening file')
            with open(self.valves.FILE_INPUT, 'r') as file:
                data = json.load(file)
                print('File successfully opened')
                return data
        except FileNotFoundError:
            raise Exception(f"File not found: {self.valves.FILE_INPUT}")
        except PermissionError:
            raise Exception(f"Permission denied for file: {self.valves.FILE_INPUT}")
        except Exception as e:
            raise Exception(f"Error opening file: {e}")

    def clear_embeddings(self):
        try:
            training_data = self.vn.get_training_data()
            if training_data is not None and 'id' in training_data:
                for row_id in training_data['id']:
                    self.vn.remove_training_data(id=row_id)
                print("Successfully removed embeddings")
            else:
                print("No embeddings to remove.")
        except Exception as e:
            print(f"Failed to clear embeddings: {e}")

    def execute_training(self):
        try:
            data = self.open_json_file()
            training_data = self.vn.get_training_data()

            if training_data is not None:
                self.clear_embeddings()

            df_information_schema = self.vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
            plan = self.vn.get_training_plan_generic(df_information_schema)

            self.vn.train(
                ddl=data['ddl'],
                documentation=data['documentation'],
                plan=plan
            )

            # Loop through question_sql_pairs and add them
            for pair in data.get('question_sql_pairs', []):
                print(f"Adding question: {pair['question']}")
                self.vn.add_question_sql(
                    question=pair['question'],
                    sql=pair['sql']
                )
            print("Training completed successfully.")
        except Exception as e:
            print(f"Failed to train data: {e}")

    def check_if_trained(self):
        try:
            # Check if embeddings exist
            training_data = self.vn.get_training_data()
            if training_data is not None and len(training_data.get('id', [])) > 0:
                return True
            else:
                return False
        except Exception as e:
            print(f"Failed to check training status: {e}")
            return False
        
        
    def directly_ask_model(self, body, user_message):
        r = requests.post(
                url=f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": self.valves.MODEL,
                    "messages": [
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
                },
                stream=False,
        )
        r.raise_for_status()

        response = r.json()
        content = response["choices"][0]["message"]["content"]
        output = response["choices"][0]
        
        print('content', content)
        print('output', output)
        
        return output

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body:dict ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        # If title generation is requested, skip the function calling filter
        print('***********************************body',body)
        if 'metadata' in body:
            print('a')
            metadata = body['metadata']
            if 'task' in metadata:
                print('aa')
                if 'title_generation' == metadata['task']:
                    print('aaa')
                    return  self.directly_ask_model( body, user_message)
                
        if body.get("title", False):
            print('bbb')
            return self.directly_ask_model( body, user_message)
        
        try:
            if self.vn is None:
                print('Error: Vanna is not connected.')
                return "Error: Vanna is not connected."
            # Generate the SQL, execute it, and create a plot
            sql, result, plot = self.vn.ask(question=user_message, allow_llm_to_see_data =True)

            # Format the output string
            formatted_sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
            sql_query = generate_sql_dropdown(formatted_sql)
            markdown_table = tabulate(result, headers='keys', tablefmt='pipe')
            response = sql_query + '\n\n' + markdown_table + '\n\n'
            
            if plot:
                # Convert the Plotly figure to a PNG image
                img_bytes = plot.to_image(format="png")
                # Encode the image in Base64
                import base64
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                # Embed the image in the Markdown response
                response += f'![Visualization](data:image/png;base64,{img_base64})\n'

            return response

        except Exception as e:
            # If anything goes wrong, return an error message
            print(f"Error occurred: {e}")
            return "Unable to provide a response. This issue is being logged and will be used to improve Hey Corona."

