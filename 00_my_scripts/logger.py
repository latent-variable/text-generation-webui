import os
import json
import logging
from pymongo import MongoClient, ASCENDING

from properties import PROPERTIES


MONGO_HOST = PROPERTIES['mongo_host'] 

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # If the message is a dictionary, use it directly
        if isinstance(record.msg, dict):
            log_record = record.msg
            # Add default logging attributes
            log_record.update({
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname,
                'name': record.name
            })
        else:
            log_record = {
                'timestamp': self.formatTime(record, self.datefmt),
                'level': record.levelname,
                'name': record.name,
                'message': record.getMessage()
            }
        return json.dumps(log_record)

def get_logger(file_name = 'chat_log.json'):
    # Set up a file handler to append to the JSON file
    file_handler = logging.FileHandler(file_name, mode='a')
    file_handler.setFormatter(JsonFormatter())

    # Configure the root logger to use the file handler
    logging.basicConfig(
        level=logging.INFO,  # Set the root logger level
        handlers=[file_handler]
    )
    

def get_chatbot_collection():
    collection = None 
    if MONGO_HOST != 'None':
        client = MongoClient(MONGO_HOST)
        db = client["chatbot"] 
        collection = db["logs"]
        collection.create_index([
            ("timestamp", ASCENDING),
            ("event", ASCENDING),
            ("user_ip", ASCENDING),
            ("action", ASCENDING)
            ], unique=True)
        
    return  collection

MONGO_COLLECTION = get_chatbot_collection()

def add_new_log_message_to_mongodb(log_entry):
   if MONGO_COLLECTION:
      MONGO_COLLECTION.append(log_entry)

def logger(log_entry, level='info'):
   if level == 'info':
      logging.info(log_entry)
   if level == 'warning':
      logging.warning(log_entry)
   add_new_log_message_to_mongodb(log_entry)
