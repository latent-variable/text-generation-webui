import os
import json
import time
import base64
import random
import getpass
import requests

import whisperx
import gradio as gr
import soundfile as sf

from collections import defaultdict

from ui import CSS_FORMAT, JS_FUNC
from logger import logger
from properties import PROPERTIES
from rag import get_relavant_docs, add_relavant_docs_to_user_prompt
from server_ports import AVAILABLE_PORTS, PORT_USERS, PORT_STATUS, \
                         USER_PORTS, get_least_loaded_port

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# Global variables
RAG_PARAMS = defaultdict(dict)
HISTORY = defaultdict(dict)
CHAT_HISTORY = defaultdict(list)
USER_INTERUPPT = defaultdict(bool)

# Max token count for the assistant
MAX_TOKEN_COUNT = 32_768 # 8192

transcriber_small = None #whisperx.load_model("base.en", device='cuda', language='en')

def get_response(history, request: gr.Request):
    '''Get the response from the assistant'''
    tik = time.time()   
    user_ip = request.client.host if request is not None else 'no-user-ip'
    user_name = getpass.getuser()
    
    if user_ip in USER_PORTS:
        # The user already has a port assigned
        port = USER_PORTS[user_ip]
        if PORT_STATUS[port]:
            # The port is in use, assign the user to a different port
            port = get_least_loaded_port()
            USER_PORTS[user_ip] = port
            PORT_USERS[port].add(user_ip)
    else:
        # This is a new user, assign them the least loaded port
        port = get_least_loaded_port()
        USER_PORTS[user_ip] = port
        PORT_USERS[port].add(user_ip)
        
    # Set the port status to in use
    PORT_STATUS[port] = True
    # Prepare the request details
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",  # Make sure server knows we accept SSE
        "Cache-Control": "no-cache"
    }
    HISTORY[user_ip] = history
    user_promt = history[-1][0]

    # Get the relevant docs based on the user message
    if RAG_PARAMS[user_ip]["use_rag"]:
        rag_docs = get_relavant_docs(user_promt, n_results=RAG_PARAMS[user_ip]["rag_n_results"])
        user_promt, metas = add_relavant_docs_to_user_prompt(user_promt, rag_docs)
        RAG_PARAMS[user_ip]["rag_context"] = user_promt  # Update rag_context with user_prompt

    if user_ip not in CHAT_HISTORY: # Initialize the chat with the system prompt
        CHAT_HISTORY[user_ip].append({"role": "system", "content": PROPERTIES['system_prompt']})
    CHAT_HISTORY[user_ip].append({"role": "user", "content": user_promt})
    data = json.dumps({
    "messages": CHAT_HISTORY[user_ip],
    "mode": "instruct",
    "stream": True,
    # "instruction_template": "Mistral",
    # "skip_special_tokens": False,
    })

    response = requests.post(url, headers=headers, data=data, stream=True)
    assistant_message = ''
    history[-1][1] = ""

    # only allow the user to stop the response if the response is being processed
    USER_INTERUPPT[user_ip] = False

    token_used = '?'
    for line in response.iter_lines():
        if  USER_INTERUPPT[user_ip]:
            break
        if line.startswith(b'data:'):
            data_json = line[5:].strip()  # Remove 'data:' prefix and any leading/trailing whitespace
            try:
                parsed_data = json.loads(data_json)
                # print(parsed_data)  # Now 'parsed_data' is a Python dictionary
                if 'usage' in parsed_data:
                    print(parsed_data['usage'])
                    token_used = parsed_data['usage']['total_tokens']
                    print(f"Token used: {token_used}")
                
                    
                
                chunk = parsed_data['choices'][0]['delta']['content']
                assistant_message += chunk
                history[-1][1] += chunk
                
                yield history
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    # Add the final message if the user did not stop the response
    if not  USER_INTERUPPT[user_ip]:
        token_message = f"\n{token_used}/{MAX_TOKEN_COUNT} Tokens"
        if RAG_PARAMS[user_ip]["use_rag"] and rag_docs is not None:
            final_message = f"📖 Sources: {metas}"
        else:
            final_message = "⚠️ This response is based entirely on my training data and may not be reliable."
     
        history[-1][1] +=  f'\n\n<sub>{final_message}</sup> <sub><sub>{token_message} </sup></sub>'
        yield history

    # Reset the user interuppt flag 
    USER_INTERUPPT[user_ip] = False
    
    # Set the port status to free
    PORT_STATUS[port] = False

    reponse_time = time.time() - tik
    # Log the user ip & user and assistant interaction details
    logger({'event':'chat', 'user_ip': user_ip, 'prompt':user_promt, 'response':assistant_message,
            'token_used': token_used, 'reponse_time': reponse_time, 'port': port, 
            'use_rag':RAG_PARAMS[user_ip]["use_rag"], 'rag_n_results': RAG_PARAMS[user_ip]["rag_n_results"]})
    
    # Add the assistant message to the chat history
    CHAT_HISTORY[user_ip].append({"role": "assistant", "content": assistant_message})    

    # randomly ask for feedback 1 out of 20 times
    if random.randint(1, 10) == 1:
        gr.Info("Did you find the response helpful? Please consider giving feedback in settings! 💡")
    
    return assistant_message


def print_like_dislike(x: gr.LikeData, request: gr.Request):
    user_ip = request.client.host if request is not None else 'no-user-ip'
    logger({'event':'feedback', 'user_ip':user_ip, 'liked':x.liked, 'reponse': x.value})


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
        
    return history, None

def redo_last_prompt(history, message, request: gr.Request):
    '''Redo the last prompt'''
    user_ip = request.client.host if request is not None else 'no-user-ip'
    multimodel_textbox = {'text':'', 'files':[]}

    if len(history) > 0:
        # Check if there are at least two messages in the history
        if len(CHAT_HISTORY[user_ip]) >= 2:
            # Remove the last user and assistant messages
            CHAT_HISTORY[user_ip].pop()
            CHAT_HISTORY[user_ip].pop()

            # Get the last user message
            multimodel_textbox['text'], _ = history.pop()
        else:
            history.clear()
            logger({'event':'action', 'user_ip': user_ip, 'action':'redo-last-prompt', 
                    'message': 'Not enough messages in history to redo last prompt'}, level='warning')

    logger({'event':'action', 'user_ip': user_ip, 'action':'redo-last-prompt'})

    return history, multimodel_textbox


def reset_chat(request: gr.Request):
    '''Reset the chatbot'''
    user_ip = request.client.host if request is not None else 'no-user-ip'
    
    # Clear the chat history
    CHAT_HISTORY[user_ip].clear()
    HISTORY[user_ip].clear()
    
    # Remove the user from the port
    if user_ip in USER_PORTS:
        if USER_PORTS[user_ip] in PORT_USERS:
            PORT_USERS[USER_PORTS[user_ip]].remove(user_ip)
        del USER_PORTS[user_ip]
    
    logger({'event':'action', 'user_ip': user_ip, 'action':'reset-chatbot'})

    return gr.Chatbot([],elem_id="chatbot", bubble_full_width=False, scale=10), \
           gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload img...", show_label=False)

def load_chat_history(request: gr.Request):
    '''Load the chat history for a user if it exists'''
    user_ip = request.client.host if request is not None else 'no-user-ip'
    # Load the chat history for the user
    chat_history = None
    if user_ip in HISTORY:
        chat_history = HISTORY[user_ip]
        logger({'event':'action', 'user_ip': user_ip, 'action':'load-chat-history'})
        
    
    # Reset the RAG parameters 
    if user_ip not in RAG_PARAMS:
        RAG_PARAMS[user_ip] = {"use_rag": False, "rag_n_results": 3, "rag_context":""}

    return chat_history, RAG_PARAMS[user_ip]["use_rag"], RAG_PARAMS[user_ip]["rag_n_results"]


def stop_response_func(request: gr.Request):
    user_ip = request.client.host if request is not None else 'no-user-ip'
    # Set the flag to stop the response
    USER_INTERUPPT[user_ip] = True
    logger({'event':'action', 'user_ip': user_ip, 'action':'Stop-Response'})
    
def update_use_rag(use_rag, request: gr.Request):
    user_ip = request.client.host if request is not None else 'no-user-ip'
    RAG_PARAMS[user_ip]['use_rag'] = use_rag


def update_rag_n_results(rag_n_results, request: gr.Request):
    user_ip = request.client.host if request is not None else 'no-user-ip'
    RAG_PARAMS[user_ip]['rag_n_results'] = int(rag_n_results)
    
def do_stt(sr,y):
    sf.write('prompt.wav', y,samplerate=sr)
    segments = transcriber_small.transcribe('prompt.wav')['segments']
    results = [segment['text'] for segment in segments]
    transcription = ' '.join(results)
    return transcription

def auto_transcribe(audio, message):
    multimodal_textbox = {'text':'', 'files':[]}
    if message['text'] is not None:
        multimodal_textbox['text'] =message['text']

    if audio is None:
        return None, multimodal_textbox

    sr, y = audio
    transcription = do_stt(sr, y)

    multimodal_textbox['text'] += transcription

    return None, multimodal_textbox


def submit_feedback(user_name, text_feedback, request: gr.Request):
    if text_feedback is None or text_feedback == '':
        return None, None
    
    user_ip = request.client.host if request is not None else 'no-user-ip'
    logger({'event':'feedback', 'user_ip': user_ip, 'user_name': user_name, 'feedback': text_feedback, 'chat_history': CHAT_HISTORY[user_ip]})

    gr.Info("Thank you for the feedback!")
    return None, None

def updated_rag_context( request: gr.Request):
    user_ip = request.client.host if request is not None else 'no-user-ip'
    
    rag_context =''
    if 'rag_context' in RAG_PARAMS[user_ip]:
        rag_context = RAG_PARAMS[user_ip]['rag_context'] 
    
    return rag_context


def image_to_data_url(image_path):
    with open(image_path, "rb") as image_file:
        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
def get_ui():
    
    title = PROPERTIES['title']
    disclaimer = PROPERTIES['disclaimer']
        
    with gr.Blocks(  title=title, fill_height=True, css=CSS_FORMAT, js=JS_FUNC,delete_cache=(86400, 86400) ) as ui:
                
        image_data_url = image_to_data_url(os.path.abspath(PROPERTIES['logo']))
        with gr.Row(elem_classes="header-row"):
            # Create a container for the title and disclaimer
            with gr.Column(elem_classes="title-disclaimer-container"):
                gr.Markdown(f"<h1 class='title'>{title}</h1>")
                gr.Markdown(f"<p class='disclaimer'>{disclaimer}</p>")
            # Add the logo on the right side
            gr.Markdown(f"<div class='logo-container'><img src='{image_data_url}' alt='Logo' class='logo'></div>", elem_id="logo-container")

        
        chatbot = gr.Chatbot([],elem_id="chatbot",bubble_full_width=False,scale=10)
        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload img...", show_label=False)

        with gr.Row():
            clear_chat = gr.Button(value="🗑️ Clear Chat ",  elem_id="new_chat")
            redo_prompt = gr.Button(value="🔁 Redo Last Prompt", elem_id="retry")
            stop_response = gr.Button(value="🛑 Stop Response", elem_id="stop_chat")

        with gr.Accordion(label='Settings ⚙️', open=False): 
            with gr.Accordion(label='Use internal knowledge 📖', open=False):
                with gr.Row(): # RAG settings 
                    use_rag = gr.Checkbox(label='Enable Retrieval Augmented Generation (RAG)', value=False )
                    rag_n_results = gr.Textbox(label='Number of chunks to use in RAG', value='3')
                with gr.Row(): # RAG context textbox
                    rag_context = gr.Textbox(label='RAG Context', value='', placeholder='Rag context will be displayed here...', lines=5, autoscroll=True)
            
            with gr.Accordion(label='Feedback 💬', open=False): 
                with gr.Column(): # Chat settings 
                    text_feedback = gr.Textbox(label='Feedback', value='', placeholder='Enter feedback here...', lines=5, autoscroll=True)
                    name_user = gr.Textbox(label='Name', value='', placeholder='Enter your name here...')
                    btn_submit = gr.Button(value='Submit Feedback', elem_id='submit_feedback')
                    
            # Add the option to selected llm model from fast, balanced, smart
            model_selection = gr.Radio(label='Select Language Model', choices=['Fast', 'Balanced', 'Smart'], value='Balanced')
            
            if transcriber_small is not None:
                with gr.Accordion(label='Text-to-Speech (TTS)  🔊', open=False):
                    # Under settings 
                    with gr.Row():
                        audio = gr.Audio(label='Speak Now', sources=['microphone'], editable=False)
                        
        # Chatbot events
        chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input]) 
        bot_msg = chat_msg.then(get_response, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        bot_msg.then(updated_rag_context, [], rag_context)
        
        # Button events
        clear_chat.click(reset_chat, inputs= [], outputs=[chatbot, chat_input])
        redo_prompt.click(redo_last_prompt, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input])
        stop_response.click(stop_response_func, inputs = [], outputs=[])
            
        # Settings events
        use_rag.change(update_use_rag, use_rag, None)
        rag_n_results.change(update_rag_n_results, rag_n_results, None)
        btn_submit.click(submit_feedback, [name_user, text_feedback], [name_user, text_feedback])

        if transcriber_small is not None:
            audio.change(auto_transcribe, inputs = [audio, chat_input], outputs=[audio, chat_input], concurrency_limit=1)

        chatbot.like(print_like_dislike, None, None)

        # Read ToS from file link
        with open(os.path.join(os.path.dirname(__file__), 'ToS.txt'), 'r') as f:
            terms_of_service_link = f.read()
        
        gr.Markdown(f"Please note: By using this service, you agree to our [Terms of Service]({terms_of_service_link}). All conversations are recorded and analyzed. This Chatbot may make mistakes. Use responsibly.")

        # Reset the chatbot for that user when the page is loaded
        ui.load(load_chat_history, [], [chatbot, use_rag, rag_n_results])

    return ui


def login(email, password):
    """Custom authentication."""

    # Do stuff here
    return True, email


if __name__ == '__main__':
    ui = get_ui()
    ui.queue(default_concurrency_limit=len(AVAILABLE_PORTS))
    ui.launch(server_port=PROPERTIES['port'], inbrowser=True, server_name= '0.0.0.0')