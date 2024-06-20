import os
import json

PROPERTIES = {}
with open(os.path.join(os.path.dirname(__file__), "properties.json"), 'r') as file:
    PROPERTIES = json.load(file)