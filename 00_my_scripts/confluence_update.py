import requests

from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import ConfluenceLoader

load_dotenv(find_dotenv())

from config import CONFLUENCE_USERNAME, CONFLUENCE_API_KEY
    
# RAG API
URL = "http://127.0.0.1:5002/api/v1/"
HEADERS = {
    "Content-Type": "application/json"
}

# Confluence
CONFLUENCE_URL = r'https://test-llm.atlassian.net/wiki' 
CONFLUENCE_SPACES = ['test', 'SD']

def load_docs_from_confluence():
    
    total_docs = []
    for space in CONFLUENCE_SPACES:
        print('Loading docs from space:', space)
        loader = ConfluenceLoader(
            url=CONFLUENCE_URL,
            username= CONFLUENCE_USERNAME, #'Lino Valdovinos',
            api_key=CONFLUENCE_API_KEY,
            space_key=space,
            limit=50,
            max_pages=10,
            keep_markdown_format=False,
            include_attachments=True, # uncomment to include png, jpeg, ..
            confluence_kwargs = {'verify_ssl': False}
        )
        
        docs = loader.load()
        for i in range(len(docs)):
            docs[i].metadata['space'] = space
   
        total_docs.extend(docs)
        
    print('Total docs:', len(total_docs))
        
    return total_docs


def GET_docs(prompt, n_results=5):

    data = {
        "mode": "instruct",
        "search_strings": [prompt],
        "n_results": n_results,
        'max_token_count': 2000,
    }
    print(data)

    response = requests.post(URL +'get', headers=HEADERS, json=data, verify=False)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return None
    

    print(response.json().keys())

    return response.json()



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
def add_confluence_docs_to_corpus(docs):

    for i, doc in enumerate(docs):
        print('Adding doc', i)
        corpus = doc.page_content
        metadatas = doc.metadata
     
        ADD_Docs(corpus, metadatas=metadatas)

# reset the database
def clear_DB():
    print("Clearing the database")
    print(URL +'clear')
    response = requests.delete(URL +'clear', headers=HEADERS, verify=False)
    print(response.json())
    
if __name__ == "__main__":
    clear_DB()
    docs = load_docs_from_confluence()
    add_confluence_docs_to_corpus(docs, )
    prompt = " the most notable observation with LLAMA3-8B under LoRA-FT quantization !c"
    print(GET_docs(prompt))

    

    