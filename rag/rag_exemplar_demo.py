# Imports
from llama_index import (
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from model_context import get_anyscale_context
from templates import custom_template

# Select Model
service_context = get_anyscale_context()
set_global_service_context(service_context)

# Load embedding store
storage_context = StorageContext.from_defaults(persist_dir="vector-db-all")
index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context
)

# Query
query_engine = index.as_query_engine(
    text_qa_template=custom_template,
    similarity_top_k=2,
)

response = query_engine.query("all birds can fly")
print("Statement to evaluate: all birds can fly\n")
print(response, "\n")
