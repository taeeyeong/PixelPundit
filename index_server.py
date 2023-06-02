import os
import pickle

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage

index = None
stored_docs = {}
lock = Lock()

index_name = "./saved_index"
pkl_name = 'stored_documents.pkl'

def initialize_index():
    """Create a new global index, or load one from the pre-set path"""
    global index, stored_docs
    
    service_context = ServiceContext.from_defaults(chunk_size_limit=512)
    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
        else:
            index = GPTVectorStoreIndex([], service_context=service_context)
            index.storage_context.persist(persist_dir=index_name)
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                stored_docs = pickle.load(f)
                
def query_index(query_text):
    """Query the global index"""
    global index
    response = index.as_query_engine().query(query_text)
    return response

def insert_into_index(doc_file_path, doc_id=None):
    """Insert new document into global index"""
    global index, stored_docs
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None: