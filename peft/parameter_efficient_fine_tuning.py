from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm



device = "cuda"




print("\n\n")
print("\n=== Defining PEFT config ===") # Follow this tutorial: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning

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




print("\n=== Loading model and tokenizer ===")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")




print("\n=== Loading dataset ===")

text_column = "text_input"
label_column = "text_label"
dataset = load_dataset("../exemplars-raw") # switch to instantiations-raw after getting peft to function

dataset = dataset.map(
    lambda x: {
        text_column: ["All " + i.lower() for i in x["generic_new"]] + ["Not all " + i.lower() for i in x["generic_new"]],
        label_column: ["False, " + i.lower() for i in x["exemplar"]] + ["True, " + i.lower() for i in x["exemplar"]]
    },
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names
)

print("\tDataset example rows")
print("\t\t", dataset["train"][0])
print("\t\t", dataset["train"][-1])






print("\n=== Set up the dataset tokenizer ===")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def preprocess_function(examples):
    batch_size = len(examples[text_column])

    # Tokenize the input text and labels
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    
    # For each example in a batch, pad the labels with the tokernizers pad_token_id
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)

        # Concatenate the input text and labels into the model_inputs.
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids

        # Create a separate attention mask for labels and model_inputs.
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    
    # Loop through each example in the batch again to pad the input ids, labels, and attention mask to the max_length and convert them to PyTorch tensors.
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs





print("\n=== Preprocess dataset ===")

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)




print("\n=== Create DataLoader ===")

train_dataset = processed_datasets["train"]
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

# eval_dataset = processed_datasets["test"]
# eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)






print("\n=== Initialize model ===")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)





print("\n=== Training ===")


