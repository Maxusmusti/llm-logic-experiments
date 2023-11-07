# Imports
from llama_index import (
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from model_context import get_anyscale_context
from templates import custom_template

import llama_index

llama_index.set_global_handler("simple")

# Select Model
service_context = get_anyscale_context()
set_global_service_context(service_context)

# Load data
storage_context = StorageContext.from_defaults(persist_dir="vector-db-refined")
index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context
)
print(type(index.vector_store))

# Query
query_engine = index.as_query_engine(
    text_qa_template=custom_template,
    # verbose=True,
    # streaming=True,
)

# response = query_engine.query("If a penguin is a bird, can penguins fly?")
# print(response, "\n")

response = query_engine.query("Batteries used in scooters are parts of automobiles")
print(response, "\n")

# response = query_engine.query("All birds eat large quantities of food")
# print(response, "\n")
