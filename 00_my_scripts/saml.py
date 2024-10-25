from flask import Flask, request
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.utils import OneLogin_Saml2_Utils

app = Flask(__name__)


# SAML settings
saml_settings = {
    "strict": True,
    "debug": False,
    "sp": {
        "entityId": "SP_ENTITY_ID",
        "assertionConsumerService": {
            "url": "ACS_URL",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        },
        "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
        "x509cert": "",
        "privateKey": ""
    },
    "idp": {
        "entityId": "IDP_ENTITY_ID",
        "singleSignOnService": {
            "url": "SSO_URL",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "x509cert": "IDP_CERTIFICATE"
    }
}

def prepare_request():
    return {
        'https': 'on' if 'https' in request.url else 'off',
        'http_host': request.host,
        'server_port': request.server_port,
        'script_name': request.path,
        'get_data': request.args.copy(),
        'post_data': request.form.copy(),
        'query_string': request.query_string
    }

@app.route('/sso', methods=['POST'])
def sso():
    # Create a SAML auth instance
    req = prepare_request()
    auth = OneLogin_Saml2_Auth(req, saml_settings)

    # Initiate SAML authentication
    auth.login()

    # After successful authentication, retrieve the user's attributes
    attributes = auth.get_attributes()
    certificate = attributes.get('certificate')

    # Do something with the certificate...

    return 'SSO completed'
    
    
    
    
if __name__ == "__main__":
    app.run(debug=True)



# # Create a SAML auth instance
# req = prepare_request()  # You need to implement this function
# auth = OneLogin_Saml2_Auth(req, saml_settings)

# # Initiate SAML authentication
# auth.login()

# # After successful authentication, retrieve the user's attributes
# attributes = auth.get_attributes()
# certificate = attributes.get('certificate')