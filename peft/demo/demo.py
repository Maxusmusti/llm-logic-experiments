from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import pandas

"""
This is almost the same as the peft/parameter_efficient_fine_tuning.py script except it is stripped down to only a short demo of model4 a.k.a PEFT(8) in our paper.
Just like the peft/parameter_efficient_fine_tuning.py script, this script draws inspiration from available Hugging Face documentation: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning 
See the peft/parameter_efficient_fine_tuning.py script for additional "START" and "END" references.
"""

print("\n=== Define PEFT config ===")
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, # length of tokens that is added to the prompt
    prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
    tokenizer_name_or_path=model_name_or_path,
)

model_load_dir = './saved_models/model4' # where to load model checkpoints from
max_length = 64
lr = 3e-2
num_epochs = 1
batch_size = 12

print("\n=== Initialize model ===")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config) # add PEFT pieces to the LLM
model.print_trainable_parameters()

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n=== Load dataset ===")
df = pandas.read_csv("./demo/data.csv")
datasetp = Dataset.from_pandas(df)
dataset = DatasetDict()
dataset["test"] = datasetp
print(dataset)
print(dataset["test"][0])

# Create data loader for evaluation
text_column = "text_input"
label_column = "text_label"

def create_evaluation_dataset(examples):
    # Define and tokenize the query
    query = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(query)
    # Loop through each example in the batch again to pad the input ids and attention mask to the max_length and convert them to PyTorch tensors.
    # START: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
    for i in range(len(examples[text_column])):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    # END: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
    return model_inputs

# Apply preprocess function to dataset
accelerator = Accelerator()
with accelerator.main_process_first():
    evaluation_dataset = dataset["test"].map(
        create_evaluation_dataset,
        batched=True,
        num_proc=1,
        remove_columns=dataset["test"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
accelerator.wait_for_everyone()
evaluation_dataset_dataloader = DataLoader(evaluation_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

print("\n=== Move model to accelerator to handle device placement ===")
# START: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(evaluation_dataset_dataloader) * num_epochs),
)
# END: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
model, evaluation_dataset_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, evaluation_dataset_dataloader, optimizer, lr_scheduler
)

print("\n=== Loading Model ===")
print("Loading model from", model_load_dir)
accelerator.load_state(input_dir=model_load_dir)

print("\n=== Evaluate model ===")
eval_preds = []
for step, batch in enumerate(tqdm(evaluation_dataset_dataloader)):
    batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=30, eos_token_id=3)
    preds = outputs[:, max_length:].detach().cpu().numpy()
    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

for pred, original in zip(eval_preds, dataset["test"]):
    print("<QUERY>\t\t\t", original[text_column].strip())
    if '.' in pred:
        period_index = pred.index('.')
        pred = pred[:period_index+1]
    print("<MODEL OUTPUT>\t\t", pred.strip())
    print("<EXPECTED OUTPUT>\t", original[label_column])
    print()
