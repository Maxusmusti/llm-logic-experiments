from transformers import LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cpu"    #"cuda"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, #ALSO MAYBE QUESTION ANSWER
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, #EXPERIMENTATION VARIABLE
    prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
    tokenizer_name_or_path=model_name_or_path,
)

text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8

model = LlamaForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


