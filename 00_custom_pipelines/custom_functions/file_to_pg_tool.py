from typing import List
import glob
import pandas as pd
import numpy as np


class Tools:
    def __init__(self):
        # If set to true it will prevent default RAG pipeline
        self.file_handler = True
        self.citation = True
        pass

    def __init__(self):
        pass

    def find_file(self, __files__):
        file_name = __files__[0]['file']['meta']['name']
        dir = "/app/backend/data/uploads"
        pattern = f"{dir}/*{file_name}"
        return glob.glob(pattern)
    
    def get_file_ext(self, file_name):
        if '.' in file_name:
            ext = '.' + file_name.rsplit('.', 1)[-1]
        else:
            ext = ''
        return ext

    def file_is_tabular(self, file_name):
        TABULAR_EXTS = {'.csv', '.xls', '.xlsx'}        
        ext = self.get_file_ext(file_name)

        return ext in TABULAR_EXTS

    def convert_file_df(self, file_path):
        file_to_df = {
            '.csv': pd.read_csv,
            '.xls': pd.read_excel,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json
        }
        ext = self.get_file_ext(file_path)
        try:
            df = file_to_df[ext](file_path)
            return df
        except:
            print("Unsupported file extension")

    def insert_into_pg(self, df):
        
        pass

    def insert_into_training_json(self):
        pass

    def get_files(self, __files__: List[dict] = []) -> str:
        """
        Get the files

        """
        file_path = self.find_file(__files__)
        file_name = __files__[0]['file']['meta']['name']
        if (self.file_is_tabular(file_name)):
            df = self.convert_file_df(file_path)
            self.insert_into_pg(df)
            self.insert_into_training_json(df)
        else:
            raise ValueError(f"Unsupported file extension")

        print('-------------------\n' , __files__, '-------------------\n')
        # return (
        #     """Show the file content directly using: `/api/v1/files/{file_id}/content`
        #         If the file is video content render the video directly using the following template: {{VIDEO_FILE_ID_[file_id]}}
        #         If the file is html file render the html directly as iframe using the following template: {{HTML_FILE_ID_[file_id]}}"""
        #     + f"Files: {str(__files__)}"
        # )

        return(f"Files: {str(__files__)}")
    

files = [{'type': 'file', 
            'file': {'id': '824072e7-2365-47e0-a470-f515e01c65cb', 
                    'user_id': '1ce6eaeb-d64f-438f-a929-61bae186f013', 
                    'hash': 'f623d68602119ce00a266ff369f4518e6f05467b3fe2051dd829f1ad7f3775c8', 
                    'filename': '824072e7-2365-47e0-a470-f515e01c65cb_test.html', 
                    'data': {'content': '\n\n\n\n\nModern Web Page\n\n\n\n\nWelcome to My Modern Web Page\nThis is a simple page designed with modern HTML and CSS techniques. Enjoy the sleek and smooth design!\nLearn More\n\n\nMade with ❤️ by Your Name\n\n\n\n'}, 
                    'meta': {'name': 'test.html', 
                            'content_type': 'text/html', 
                            'size': 2916, 
                            'path': '/app/backend/data/uploads/824072e7-2365-47e0-a470-f515e01c65cb_test.html', 
                            'collection_name': 'file-824072e7-2365-47e0-a470-f515e01c65cb'}, 
                    'created_at': 1731716955, 
                    'updated_at': 1731716955}, 
            'id': '824072e7-2365-47e0-a470-f515e01c65cb', 
            'url': '/api/v1/files/824072e7-2365-47e0-a470-f515e01c65cb', 
            'name': 'test.html', 
            'collection_name': 'file-824072e7-2365-47e0-a470-f515e01c65cb', 
            'status': 'uploaded', 
            'size': 2916, 
            'error': ''}]
    
file = files[0]

# print(file['file']['meta']['name'])
# file_name  = file['file']['meta']['name']
# file_name = 'cool.html'
# print(file_name)

# def find_file(file_name):
#     dir = "hey_corona/custom_pipelines"
#     pattern = f"{dir}/*{file_name}"
#     return glob.glob(pattern)

# path = find_file('_router.py')
# print(f"Here is the file path:\n{path}")

# def file_is_tabular(file_name):
#     TABULAR_EXTS = {'.csv', '.xls', '.xlsx'}
#     # Use rsplit to avoid regex overhead
#     if '.' in file_name:
#         ext = '.' + file_name.rsplit('.', 1)[-1]
#     else:
#         ext = ''
#     return ext in TABULAR_EXTS

# print(file_is_tabular(file_name))

def get_file_ext(file):
    if '.' in file:
        ext = '.' + file.rsplit('.', 1)[-1]
    else:
        ext = ''
    return ext

def convert_file_df(file_path):
    file_to_df = {
        '.csv': pd.read_csv,
        '.xls': pd.read_excel,
        '.xlsx': pd.read_excel,
        '.json': pd.read_json
    }
    ext = get_file_ext(file_path)
    try:
        df = file_to_df[ext](file_path)  # Pass the file object instead of path
        return df
    except Exception as e:
        print(f"Error: {e}")

file_path = "hey_corona\\training\\training.json"
ext = get_file_ext(file_path)
print(f"ext: {ext}")
df = convert_file_df(file_path)

print(f"Here is the dataframe of training.json:\n{df}")
