import os
from fastapi import FastAPI, Depends, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
import uvicorn
import gradio as gr
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from urllib.parse import urlparse

app = FastAPI()

SECRET_KEY = os.environ.get('SECRET_KEY') or "a_very_secret_key"
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

def init_saml_auth(req: dict):
    auth = OneLogin_Saml2_Auth(req, custom_base_path=os.path.join(os.getcwd(), 'saml'))
    return auth

async def prepare_flask_request(request: Request):
    url_data = urlparse(str(request.url))
    return {
        'https': 'on' if request.url.scheme == 'https' else 'off',
        'http_host': request.url.hostname,
        'server_port': url_data.port,
        'script_name': request.url.path,
        'get_data': request.query_params,
        'post_data': await request.form()
    }

# Dependency to get the current user
def get_user(request: Request):
    user = request.session.get('user')
    if user:
        return user['name']
    return None

@app.get('/')
def public(user: dict = Depends(get_user)):
    if user:
        return RedirectResponse(url='/gradio')
    else:
        return RedirectResponse(url='/login-demo')

@app.get('/logout')
async def logout(request: Request):
    req = await prepare_flask_request(request)
    auth = init_saml_auth(req)
    request.session.pop('user', None)
    return RedirectResponse(url=auth.logout())

@app.get('/login')
async def login(request: Request):
    req = await prepare_flask_request(request)
    auth = init_saml_auth(req)
    return RedirectResponse(url=auth.login())

@app.post('/auth')
async def auth(request: Request):
    req = await prepare_flask_request(request)
    auth = init_saml_auth(req)
    auth.process_response()
    errors = auth.get_errors()
    if len(errors) == 0:
        request.session['user'] = auth.get_attributes()
        return RedirectResponse(url='/')
    else:
        return RedirectResponse(url='/')

with gr.Blocks() as login_demo:
    gr.Button("Login", link="/login")

app = gr.mount_gradio_app(app, login_demo, path="/login-demo")

def greet(request: gr.Request):
    return f"Welcome to Gradio, {request.username}"

with gr.Blocks() as main_demo:
    m = gr.Markdown("Welcome to Gradio!")
    gr.Button("Logout", link="/logout")
    main_demo.load(greet, None, m)

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)

if __name__ == '__main__':
    uvicorn.run(app)
