@echo off

rem Set environment variables
set WEBUI_NAME=Hey Corona
set CUSTOM_NAME=Hey Corona
set WEBUI_AUTH=True
set DEFAULT_USER_ROLE="user"
set ENABLE_SIGNUP=True
set USE_CUDA_DOCKER=True
set ADMIN_EMAIL="lino.valdovinos.civ@us.navy.mil"
set DATA_DIR=./00_my_data
set ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION=True
set SAML_METADATA_FILE="./saml/metadata.xml"
set OAUTH_CLIENT_ID="open-webui-client"
set OAUTH_CLIENT_SECRET="KVQNjapG36muPocrFN8FVhACL9EUMKDR"
set OPENID_PROVIDER_URL="http://keycloak:8080/realms/open-webui-realm/.well-known/openid-configuration"
set OAUTH_SCOPES="admin@webui.com"
set ENABLE_OAUTH_SIGNUP=True

rem Activate conda environment
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"

rem Run the application
CALL open-webui serve

rem Keep the command prompt open
cmd /k
