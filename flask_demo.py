import os 
from dotenv import load_dotenv
load_dotenv()
from llama_index import(
    StorageContext
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

index = None

def initialize_index():
    global index
    storage_context = StorageContext.from_defaults()
    if os.path.exists(index_dir):
        index = load_index_from_