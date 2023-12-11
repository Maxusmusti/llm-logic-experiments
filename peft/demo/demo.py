from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import Dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, load_checkpoint_in_model
import pandas


print("\n=== Define PEFT config ===")

model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, # length of tokens that is added to the prompt
    prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
    tokenizer_name_or_path=model_name_or_path,
)


train = False # determines whether to train and save a new model or load a saved model
evaluate_performance = True # determines whether to run the evaluation script at the end of the script to measure model accuracy

model_save_dir = './saved_models/model4' # where to save the model once it's trained
model_load_dir = model_save_dir # where to load model checkpoints from if one is being laoded

max_length = 64
lr = 3e-2
num_epochs = 1
batch_size = 12
test_split_size = 0.1





print("\n=== Initialize model ===")

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
model = get_peft_model(model, peft_config) # add PEFT pieces to the LLM
model.print_trainable_parameters()






print("\n=== Load dataset ===")

text_column = "text_input"
label_column = "text_label"

negatvie_csv_path = "./demo/data.csv"

# load the data into a pandas dataframe and group by generic to have only 1 instance of each generic in the dataset
df_neg = pandas.read_csv(negatvie_csv_path)

# load the dataframe into a datset
dataset_neg = Dataset.from_pandas(df_neg)

# Split dataset into train and test split before doing any data augmentation
dataset_neg = dataset_neg.train_test_split(test_size=test_split_size, seed=10)

# Negative samples data augmentation: Add "All " and "False, " and "Not all ", and "True, "
dataset_neg = dataset_neg.map(
    lambda x: {
        text_column: ["All " + i.lower() for i in x["generic"]] + ["Not all " + i.lower() for i in x["generic"]],
        label_column: ["False, " + i.lower() for i in x["exemplar"]] + ["True, " + i.lower() for i in x["exemplar"]]
    },
    batched=True,
    num_proc=1,
    remove_columns=dataset_neg["train"].column_names
)

# Combine negative and positive datasets
dataset_train = dataset_neg["train"]
dataset_test = dataset_neg["test"]
dataset = DatasetDict()
dataset["train"] = dataset_train
dataset["test"] = dataset_test

print(dataset)
print(dataset["train"][0])
print(dataset["test"][0])




# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token="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")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function(examples):
    batch_size = len(examples[text_column])

    # Tokenize the input text and labels
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]] # Format the data into an actual natural-language query for the LLM
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    
    # For each example in a batch, pad the labels with the tokernizer's pad_token_id
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]

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

# Apply preprocess function to dataset
accelerator = Accelerator()
with accelerator.main_process_first():
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
accelerator.wait_for_everyone()

# Create data loaders
train_dataset = processed_datasets["train"]
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

eval_dataset = processed_datasets["test"]
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)















print("\n=== Initialize optimizer ===")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("\n=== Initialize Learning Rate scheduler ===")

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

print("\n=== Move model to accelerator to handle device placement ===")

model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
)
accelerator.print(model)





print("\n=== Loading Model ===")

# load the model
print("Loading model from", model_load_dir)
accelerator.load_state(input_dir=model_load_dir)













if evaluate_performance:
    
    print("\n=== Evaluate model ===")

    # Create data loader for evaluation that intentionally leaves out the answer so the model can fill it in
    def create_evaluation_dataset(examples):
        batch_size = len(examples[text_column])

        # Define and tokenize the query
        query = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        model_inputs = tokenizer(query)

        # Loop through each example in the batch again to pad the input ids and attention mask to the max_length and convert them to PyTorch tensors.
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        
        return model_inputs

    evaluation_dataset = dataset["test"].map(create_evaluation_dataset, batched=True, num_proc=1)
    evaluation_dataset_dataloader = DataLoader(evaluation_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    eval_preds = []
    for step, batch in enumerate(tqdm(evaluation_dataset_dataloader)):
        batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model.generate(**batch, max_new_tokens=30, eos_token_id=3)
        preds = outputs[:, max_length:].detach().cpu().numpy()
        eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

    for pred, original in zip(eval_preds, dataset["test"]):
        print("<QUERY>\t\t\t", original[text_column].strip())
        print("<MODEL OUTPUT>\t\t", pred.strip())
        print("<EXPECTED OUTPUT>\t", original[label_column].strip())
        print()

