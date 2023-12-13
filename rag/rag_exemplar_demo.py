# Imports
from llama_index import (
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from model_context import get_anyscale_context
from templates import custom_template

# Select Model
print("Loading model context...")
service_context = get_anyscale_context()
set_global_service_context(service_context)

# Load embedding store
print("Loading RAG embeddings...")
storage_context = StorageContext.from_defaults(persist_dir="vector-db-all")
index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context
)

# Query
query_engine = index.as_query_engine(
    text_qa_template=custom_template,
    similarity_top_k=2,
)

print("\n\n-------------------------\n\n")

print("\nStatement to evaluate: not all Birds depend on vegetation.\n")
response = query_engine.query("not all Birds depend on vegetation.")
print(response, "\n")

print("\n\n-------------------------\n\n")

print("\nStatement to evaluate: not all Toucans live in rainforests.\n")
response = query_engine.query("not all Toucans live in rainforests.")
print(response, "\n")
