import gradio as gr
import ssl
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

# Predefined token for authentication
VALID_TOKEN = "your_secret_token"

def authenticate(token):
    return token == VALID_TOKEN

def protected_function(input_text):
    return f"You entered: {input_text}"

# Create the main interface
with gr.Blocks() as demo:
    gr.Markdown("# Protected Gradio App with SSL")
    
    with gr.Group():
        token_input = gr.Textbox(label="Enter Token", type="password")
        auth_button = gr.Button("Authenticate")
        auth_status = gr.Markdown("Not authenticated")

    with gr.Group() as protected_group:
        input_text = gr.Textbox(label="Enter some text")
        output_text = gr.Textbox(label="Output")
        submit_button = gr.Button("Submit")

    protected_group.visible = False

    def check_auth(token):
        if authenticate(token):
            protected_group.visible = True
            return "Authenticated successfully!"
        else:
            protected_group.visible = False
            return "Authentication failed. Please try again."

    auth_button.click(check_auth, inputs=token_input, outputs=auth_status)
    submit_button.click(protected_function, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    # Set up SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    # Create a custom HTTPS server
    class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(302)
                self.send_header('Location', 'https://localhost:7860')
                self.end_headers()
            else:
                super().do_GET()

    httpd = HTTPServer(('localhost', 8000), CustomHTTPRequestHandler)
    httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)

    # Start the HTTPS server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()

    # Launch Gradio app
    demo.launch(server_name="localhost", server_port=445, share=False)

    # Open the browser
    webbrowser.open('https://localhost:8000')

    # Keep the main thread running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down the server...")
        httpd.shutdown()
        server_thread.join()