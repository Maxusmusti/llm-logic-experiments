from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm



device = "cpu" # can change to "cuda"




print("\n\n")
print("=== Defining PEFT config ===") # Follow this tutorial: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, # could also be QUESTION_ANS
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, # length of tokens that is added to the prompt
    prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
    tokenizer_name_or_path=model_name_or_path,
)

max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8




print("=== Loading model and tokenizer ===")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")




print("=== Loading dataset ===")

generic_column = "generic_new"
exemplar_column = "exemplar"
dataset = load_dataset("../exemplars-prev")




print("=== Set up the dataset tokenizer ===")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)





'''
How to set up the dataset:

From exemplars-prev:

Input text: “All “ + lowercase(generic) + “: “ -> False + examplar
Labels: Not all + generic -> True + exemplar
'''


