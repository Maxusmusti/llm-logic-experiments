# Imports
from llama_index import SimpleDirectoryReader, VectorStoreIndex, set_global_service_context
from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.text_splitter import SentenceSplitter

from model_context import get_anyscale_context

# Select Model
service_context = get_anyscale_context()
set_global_service_context(service_context)

# Load data
from llama_hub.file.paged_csv.base import PagedCSVReader
documents = SimpleDirectoryReader(file_extractor={".csv": PagedCSVReader(encoding="utf-8")}, input_dir="./exemplars").load_data()
#documents = SimpleDirectoryReader('./exemplars').load_data()

# Parse data
#text_splitter = SentenceSplitter(separator="\n")
#def newline_splitter(text):
 #   return text.split("\n")

#parser = SentenceWindowNodeParser.from_defaults(sentence_splitter=newline_splitter, window_size=1)
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents, show_progress=True)

# Vectorize, index, and store data
index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
index.storage_context.persist(persist_dir="vector-db")
