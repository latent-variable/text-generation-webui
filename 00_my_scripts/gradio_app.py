from flask import Flask, request
import gradio as gr
from werkzeug.serving import run_simple
import ssl

app = Flask(__name__)

def authenticate():
    cert = request.environ.get('SSL_CLIENT_CERT')
    if cert:
        # Here you would validate the certificate
        return True, f"Authenticated with certificate: {cert[:30]}..."
    else:
        return False, "No client certificate provided"

def protected_function(input):
    is_authenticated, message = authenticate()
    if is_authenticated:
        return f"Hello {input}! {message}"
    else:
        return "Access denied. Please authenticate with a valid client certificate."

iface = gr.Interface(
    fn=protected_function,
    inputs="text",
    outputs="text",
    title="Protected Gradio App"
)

@app.route("/", methods=["GET", "POST"])
def gradio_interface():
    return gr.routes.App(iface).app(request)

if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain('server.crt', 'server.key')
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    ssl_context.load_verify_locations('ca.crt')

    run_simple('localhost', 7860, app, ssl_context=ssl_context)