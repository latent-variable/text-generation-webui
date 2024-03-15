import threading
import chromadb
import posthog
import torch
import math

import numpy as np
import extensions.superboogav3.parameters as parameters

from chromadb.config import Settings
# from typing import  Union, Generic, TypeVar, List
from chromadb import Documents,  Embeddings, EmbeddingFunction

from sentence_transformers import SentenceTransformer

from modules.logging_colors import logger
from modules.text_generation import encode, decode

logger.debug('Intercepting all calls to posthog.')
posthog.capture = lambda *args, **kwargs: None


class Collecter():
    def __init__(self):
        pass

    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int]):
        pass

    def get(self, search_strings: list[str], n_results: int) -> list[str]:
        pass

    def clear(self):
        pass



class Info:
    def __init__(self, start_index, text_with_context, distance, id):
        self.text_with_context = text_with_context
        self.start_index = start_index
        self.distance = distance
        self.id = id

    def calculate_distance(self, other_info):
        if parameters.get_new_dist_strategy() == parameters.DIST_MIN_STRATEGY:
            # Min
            return min(self.distance, other_info.distance)
        elif parameters.get_new_dist_strategy() == parameters.DIST_HARMONIC_STRATEGY:
            # Harmonic mean
            return 2 * (self.distance * other_info.distance) / (self.distance + other_info.distance)
        elif parameters.get_new_dist_strategy() == parameters.DIST_GEOMETRIC_STRATEGY:
            # Geometric mean
            return (self.distance * other_info.distance) ** 0.5
        elif parameters.get_new_dist_strategy() == parameters.DIST_ARITHMETIC_STRATEGY:
            # Arithmetic mean
            return (self.distance + other_info.distance) / 2
        else: # Min is default
            return min(self.distance, other_info.distance)

    def merge_with(self, other_info):
        s1 = self.text_with_context
        s2 = other_info.text_with_context
        s1_start = self.start_index
        s2_start = other_info.start_index
        
        new_dist = self.calculate_distance(other_info)

        if self.should_merge(s1, s2, s1_start, s2_start):
            if s1_start <= s2_start:
                if s1_start + len(s1) >= s2_start + len(s2):  # if s1 completely covers s2
                    return Info(s1_start, s1, new_dist, self.id)
                else:
                    overlap = max(0, s1_start + len(s1) - s2_start)
                    return Info(s1_start, s1 + s2[overlap:], new_dist, self.id)
            else:
                if s2_start + len(s2) >= s1_start + len(s1):  # if s2 completely covers s1
                    return Info(s2_start, s2, new_dist, other_info.id)
                else:
                    overlap = max(0, s2_start + len(s2) - s1_start)
                    return Info(s2_start, s2 + s1[overlap:], new_dist, other_info.id)

        return None
    
    @staticmethod
    def should_merge(s1, s2, s1_start, s2_start):
        # Check if s1 and s2 are adjacent or overlapping
        s1_end = s1_start + len(s1)
        s2_end = s2_start + len(s2)
        
        return not (s1_end < s2_start or s2_end < s1_start)

class ChromaCollector(Collecter):
    def __init__(self, embedder: EmbeddingFunction):
        super().__init__()
        self.chroma_client = chromadb.PersistentClient(path=r".chroma", settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.embedder = embedder
        self.collection = self.chroma_client.get_or_create_collection(name="context", embedding_function=self.embedder)
        self.max_ids = self.collection.count()
        logger.info(f'ChromaDB initialized with {self.max_ids} records.')

        self.id_to_info = {}
        self.lock = threading.Lock() # Locking so the server doesn't break.

    def add(self, texts: list[str], texts_with_context: list[str], starting_indices: list[int], metadatas: list[dict] = None):
        with self.lock:
            assert metadatas is None or len(metadatas) == len(texts), "metadatas must be None or have the same length as texts"
            
            if len(texts) == 0: 
                return

            new_ids = self._get_new_ids(len(texts))
            

            (non_existing_texts, non_existing_ids, non_existing_metas) = self._prepare_text(texts, new_ids, metadatas)
            print('non_existing_ids:', non_existing_ids)

            # If there are any non-existing texts, compute their embeddings all at once. Each call to embed has significant overhead.
            if non_existing_texts:
                non_existing_embeddings = self.embedder.embed(non_existing_texts).tolist()
                
                args = {'embeddings': non_existing_embeddings, 'documents': non_existing_texts, 'ids': non_existing_ids}

                if metadatas is not None: 
                    args['metadatas'] = non_existing_metas

                self.collection.add(**args)
                logger.info(f'Added {len(non_existing_embeddings)} new embeddings.')
    
    def _prepare_text(self, texts: list[str], new_ids: list[str], metadatas: list[dict]):
        non_existing_texts = []
        non_existing_ids = []
        non_existing_metas = []

        for i, text in enumerate(texts):
            id_ = new_ids[i]
            metadata = metadatas[i] if metadatas is not None else None
            
            non_existing_texts.append(text)
            non_existing_ids.append(id_)
            non_existing_metas.append(metadata)

        return (non_existing_texts, non_existing_ids, non_existing_metas)


    def _get_new_ids(self, num_new_ids: int):
        max_existing_id =  self.max_ids - 1
        self.max_ids += num_new_ids

        return [str(i + max_existing_id + 1) for i in range(num_new_ids)]

    
    def _find_min_max_start_index(self):
        max_index, min_index = 0, float('inf')
        for _, val in self.id_to_info.items():
            if val['start_index'] > max_index:
                max_index = val['start_index']
            if val['start_index'] < min_index:
                min_index = val['start_index']
        return min_index, max_index



    def _merge_infos(self, infos: list[Info]):
        merged_infos = []
        current_info = infos[0]

        for next_info in infos[1:]:
            merged = current_info.merge_with(next_info)
            if merged is not None:
                current_info = merged
            else:
                merged_infos.append(current_info)
                current_info = next_info

        merged_infos.append(current_info)
        return merged_infos


    # Main function for retrieving chunks by distance. It performs merging, time weighing, and mean filtering.
    def _get_documents_ids_distances(self, search_string: str, n_results: int):
        n_results = min(self.max_ids, n_results)
        if n_results == 0:
            return [], [], [], []
            
        result = self.collection.query(query_texts=search_string, n_results=n_results, include=['distances', "metadatas", "documents"])

        print('actual number of results', len(result['documents'][0]))
        print('actual number of ids', result['ids'][0])


        return result['documents'][0], result['ids'][0], result['distances'][0], result['metadatas'][0]
        
   

    # Get chunks by similarity
    def get(self, search_string: str, n_results: int) -> list[str]:
        with self.lock:
            documents, _, _, _ = self._get_documents_ids_distances(search_string, n_results)
            return documents
    

    # Get ids by similarity
    def get_ids(self, search_string: str, n_results: int) -> list[str]:
        with self.lock:
            _, ids, _, _ = self._get_documents_ids_distances(search_string, n_results)
            return ids
    
    
    # Cutoff token count
    def _get_documents_up_to_token_count(self, documents: list[str], max_token_count: int):
        # TODO: Move to caller; We add delimiters there which might go over the limit.
        current_token_count = 0
        return_documents = []

        for doc in documents:
            doc_tokens = encode(doc)[0]
            doc_token_count = len(doc_tokens)
            if current_token_count + doc_token_count > max_token_count:
                # If adding this document would exceed the max token count,
                # truncate the document to fit within the limit.
                remaining_tokens = max_token_count - current_token_count
                
                truncated_doc = decode(doc_tokens[:remaining_tokens], skip_special_tokens=True)
                return_documents.append(truncated_doc)
                break
            else:
                return_documents.append(doc)
                current_token_count += doc_token_count

        return return_documents
    

    # Get chunks by similarity and then sort by ids
    def get_sorted_by_ids(self, search_string: str, n_results: int, max_token_count: int, include_metadata=False) -> list[str]:
        with self.lock:
            documents, ids, _, metadatas = self._get_documents_ids_distances(search_string, n_results)
            sorted_docs = [x for _, x in sorted(zip(ids, documents))]

            sorted_metadatas = [meta for _, meta in sorted(zip(ids, metadatas))]

            if include_metadata:
                return self._get_documents_up_to_token_count(sorted_docs, max_token_count), sorted_metadatas
            else:
                return self._get_documents_up_to_token_count(sorted_docs, max_token_count)
    
    # Get chunks by similarity and then sort by distance (lowest distance is last).
    def get_sorted_by_dist(self, search_strings: str, n_results: int, max_token_count: int, include_metadata=False) -> list[str]:
        with self.lock:
            documents, _, distances, metedatas = self._get_documents_ids_distances(search_strings, n_results)
            sorted_docs = [doc for doc, _ in sorted(zip(documents, distances), key=lambda x: x[1])] # sorted lowest -> highest
            sorted_metadatas = [meta for meta, _ in sorted(zip(metedatas, distances), key=lambda x: x[1])]
            
            # If a document is truncated or competely skipped, it would be with high distance.
            return_documents = self._get_documents_up_to_token_count(sorted_docs, max_token_count)
            return_documents.reverse() # highest -> lowest
            
            if include_metadata:
                return return_documents, sorted_metadatas
            else:
                return return_documents
    

    def delete(self, ids_to_delete: list[str], where: dict):
        with self.lock:
            ids_to_delete = self.collection.get(ids=ids_to_delete, where=where)['ids']
            self.collection.delete(ids=ids_to_delete, where=where)
            logger.info(f'Successfully deleted {len(ids_to_delete)} records from chromaDB.')


    def clear(self):
        with self.lock:
            self.chroma_client.reset()
            self.collection = self.chroma_client.create_collection("context", embedding_function=self.embedder.embed)
            self.max_ids = 0

            logger.info('Successfully cleared all records and reset chromaDB.')


class SentenceTransformerEmbedder(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        return self.model.encode(input)
    
    def embed(self, input: Documents) -> Embeddings:
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        return self.model.encode(input)
    

def make_collector():
    return ChromaCollector(SentenceTransformerEmbedder())