"""
This module contains common functions across multiple other modules.
"""
import os
import extensions.superboogav3.parameters as parameters

# Create the context using the prefix + data_separator + postfix from parameters.
def create_context_text(results):
    if results is None or len(results) == 0:
        results = ["\nNo matching documents found.\n"]
    
    context = parameters.get_prefix() + parameters.get_data_separator().join(results) + parameters.get_postfix()

    return context


# Create metadata with the specified source
def create_metadata_source(source: str):
    return {'source': source}