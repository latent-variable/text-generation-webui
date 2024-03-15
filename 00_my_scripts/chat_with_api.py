import requests

# URL for the cloud API
cloud_URL = 'https://alternate-ownership-rooms-levels.trycloudflare.com/v1/chat/completions'
local_URL = "http://0.0.0.0:5000/api/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json"
}

HISTORY = []

def web_llm_call(prompt, conversation=False, use_cloud=True):
    if conversation:
        HISTORY.append({"role": "user", "content": prompt})
    else:
        HISTORY = [{"role": "user", "content": prompt}]
    data = {
        "mode": "instruct",
        "character":"Assistant",
        "messages": HISTORY
    }
    if use_cloud:
        URL = cloud_URL
    else:
        URL = local_URL
    # Call cloud API 
    response = requests.post(URL, headers=HEADERS, json=data, verify=False)

    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return None
    

    print(response.json())

    assistant_message = response.json()['choices'][0]['message']['content']
    HISTORY.append({"role": "assistant", "content": assistant_message})
    print(assistant_message)
    return assistant_message




if __name__=="__main__":
    prompt = "Methods of levying anti-dumping duties From January 1, 2023? !c"
    web_llm_call(prompt, conversation=False)


