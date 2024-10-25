# Imports
# Env var
import os
from dotenv import load_dotenv, find_dotenv

# Env variables
_ = load_dotenv(find_dotenv())

CONFLUENCE_API_KEY = os.environ['CONFLUENCE_PRIVATE_API_KEY']
# https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/
# Hint: space_key and page_id can both be found in the URL of a page in Confluence
# https://yoursite.atlassian.com/wiki/spaces/<space_key>/pages/<page_id>
CONFLUENCE_USERNAME = os.environ['EMAIL_ADRESS']
