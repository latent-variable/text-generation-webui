import requests
import pandas as pd 

URL = "http://127.0.0.1:5002/api/v1/"


HEADERS = {
    "Content-Type": "application/json"
}

def GET_docs(prompt, n_results=5):

    data = {
        "mode": "instruct",
        "search_strings": [prompt],
        "n_results": n_results,
        'max_token_count': 2000,
        # 'basic_query': 'true',
        # "messages": HISTORY
    }
    print(data)

    response = requests.post(URL +'get', headers=HEADERS, json=data, verify=False)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return None
    

    print(response.json().keys())

    return response.json()
    # assistant_message = response.json()['choices'][0]['message']['content']
    # HISTORY.append({"role": "assistant", "content": assistant_message})
    # print(assistant_message)
    # return assistant_message


def ADD_Docs(corpus, metadatas={}, clear_before_adding=False):
    data = {
        "corpus": corpus, 
        "clear_before_adding": clear_before_adding,
        "metadata": metadatas
    }
    # print(data)
    response = requests.post(URL +'add', headers=HEADERS, json=data, verify=False)
    print(response.json())


# Add the data to the corpus
def add_csv_data_to_corpus(file_path, context_key= "text"):
    # Read the csv file
    data = pd.read_csv(file_path)
    
    # Convert the data to a list of dictionaries
    data = data.to_dict(orient='records')
    
    # # Loop through the data
    for i, row in enumerate(data):
        # if i in [0, 1]:
        corpus = row[context_key]
        metadatas = {}
        for key in row.keys():
            if key != context_key:
                metadatas[key] = row[key]

        ADD_Docs(corpus, metadatas=metadatas)

# reset the database
def clear_DB():
    print("Clearing the database")
    print(URL +'clear')
    response = requests.delete(URL +'clear', headers=HEADERS, verify=False)
    print(response.json())
    
if __name__ == "__main__":
    clear_DB()
    
    # key = "text"
    # file_path = r"./data/dataset.csv"

    key = "translated_content"
    file_path = r"./data/mofcom_translated_cleaned_summarized_v2.csv"
    add_csv_data_to_corpus(file_path, context_key=key )


    prompt = "the Ministry of Commerce (MOFCOM) issued what Announcement? !c"
    # prompt = " tell me about cats"
    print(GET_docs(prompt))

    

    