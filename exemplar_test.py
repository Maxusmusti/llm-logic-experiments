# Imports
from llama_index import SimpleDirectoryReader, VectorStoreIndex, set_global_service_context
from llama_index.node_parser import SimpleNodeParser
from model_context import get_stablelm_context, get_falcon_context, get_llama_context
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.prompts import Prompt, PromptType

import llama_index
llama_index.set_global_handler("simple")

# Select Model
service_context = get_falcon_context()
set_global_service_context(service_context)

# Load data
documents = SimpleDirectoryReader('./exemplars').load_data()

# Parse data
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# Index data
index = VectorStoreIndex(nodes, service_context=service_context)
print(type(index.vector_store))

# Custom Template
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Below is a generic statement that is usually true, followed by a list of true and false examples:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the below question carefully, making sure that it does not contradict any above examples (even if that means saying the generic statement is false)\n"
    "Query: {query_str}\n"
    "Answer: "
)
custom_template = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

# Query
query_engine = index.as_query_engine(
    text_qa_template=custom_template,
    #verbose=True,
    #streaming=True,
)

response = query_engine.query("If a penguin is a bird, can penguins fly?")
print(response, "\n")
