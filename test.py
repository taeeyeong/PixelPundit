from langchain import OpenAI
from llama_index import (
    download_loader,
    SimpleDirectoryReader,
    LLMPredictor,
    GPTVectorStoreIndex,
    GPTKeywordTableIndex,
    PromptHelper,
    ServiceContext)
from langchain import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
documents = loader.load_data()


# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

max_input_size = 4095
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTKeywordTableIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("How to estimate pose parameters?")
print(response)
