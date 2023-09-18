# Imports
from llama_index import SimpleDirectoryReader, VectorStoreIndex, set_global_service_context
from llama_index.node_parser import SimpleNodeParser

from model_context import get_falcon_context

# Select Model
service_context = get_falcon_context()
set_global_service_context(service_context)

# Load data
documents = SimpleDirectoryReader('./exemplars').load_data()

# Parse data
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Vectorize, index, and store data
index = VectorStoreIndex(nodes, service_context=service_context)
index.storage_context.persist(persist_dir="vector-db")
