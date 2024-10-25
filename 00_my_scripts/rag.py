import requests

# Get the relevant documents based on the user prompts
def get_relavant_docs(prompt, n_results=3):
    
    URL = "http://127.0.0.1:5002/api/v1/"
    HEADERS = {
        "Content-Type": "application/json"
    }

    data = {
        "mode": "instruct",
        "search_strings": [prompt],
        "n_results": n_results,
        'max_token_count': 2000,
    }

    response = requests.post(URL +'get', headers=HEADERS, json=data, verify=False)
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return None
    
    results = response.json()
    if len(results['results']) == 0:
        return None
    else:
        return results
  
def add_relavant_docs_to_user_prompt(user_promt, docs):
    if docs is None or len(docs) == 0:
        prompt_engineering ="""
        There are no relevant documents for the user's prompt. 
        Let the user know that you were not able to find relevant information for their question, 
        but try to help them to the best of your abilities.
        """
        updated_prompt = f'{prompt_engineering} \n User prompt: {user_promt}'
        return updated_prompt, None
    
    doc_start = "\n\n<<document chunk>>\n\n"
    doc_end = "\n\n<<document end>>\n\n"

    new_user_promt =  ''
    for doc in docs['results']:
        new_user_promt += doc_start + doc 
    
    meta_set = set([])
    metas = ''
    for meta in docs['meta']:
        if meta['source'] in meta_set:
            continue
        meta_set.add(meta['source'])
        metas += f"\n{meta['title']} - {meta['source']}"
    
    prompt_engineering = """
    Please use the provided documents to generate a response to the user's prompt. 
    """
    updated_prompt = f'{new_user_promt} {doc_end} {prompt_engineering} \n User prompt: {user_promt}'
    return updated_prompt , metas