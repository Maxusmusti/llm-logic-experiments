# Imports
from collections import defaultdict
from llama_index import (
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from model_context import get_anyscale_context
from templates import blank_template, custom_template
import csv
from tqdm import tqdm

# DEBUG LOGS
# import llama_index
# llama_index.set_global_handler("simple")

# Select Model
print("Loading model context...")
service_context = get_anyscale_context()
set_global_service_context(service_context)

# Load embedded data for RAG
print("Loading RAG embeddings...")
storage_context = StorageContext.from_defaults(persist_dir="vector-db")
index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context
)
# Assemble Query Engine
query_engine = index.as_query_engine(
    text_qa_template=custom_template,
    # verbose=True,
    # streaming=True,
)

# Load evaluation data
print("Loading evaluation data...")
labeled_data = defaultdict(list)
with open("exemplars/exceptions.onlyValid.csv", "r") as full_data:
    data_reader = csv.DictReader(full_data)
    for sample in data_reader:
        labeled_data[sample["generic_new"]].append(sample["exemplar"])
print(f"{len(labeled_data)} generics loaded!")
generics = list(labeled_data.keys())

# Evaluation Loop
print("Beginning evaluation:")
tie = 0
loss = 0
win = 0
for i in tqdm(range(len(generics)), desc="Generic evaluation process"):
    sample = generics[i]
    for ext in ["All", "Not all"]:
        prompt = ext + " " + sample
        response = query_engine.query(prompt)
        false_count = str(response).lower().count("false")
        true_count = str(response).lower().count("true")

        if ext == "All":
            good = false_count
            bad = true_count
        elif ext == "Not all":
            good = true_count
            bad = false_count

        if good > bad:
            win += 1
        elif bad > good:
            loss += 1
        else:
            tie += 1

print("Wins: ", win)
print("Ties: ", tie)
print("Loss: ", loss)
