from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import Dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, load_checkpoint_in_model
import pandas
from util.accuracy_evaluation import *

# This script draws inspiration from available Hugging Face documentation: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning 
def main():

    print("\n=== Define PEFT config ===")

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=4, # length of tokens that is added to the prompt
        prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
        tokenizer_name_or_path=model_name_or_path,
    )


    debug = True # determines how much of dataset to use. debug=True means only debug_size% of data is used and only 1 epoch
    debug_size = 25

    train = False # determines whether to train and save a new model or load a saved model
    evaluate_performance = True # determines whether to run the evaluation script at the end of the script to measure model accuracy

    model_save_dir = './saved_models/model6' # where to save the model once it's trained
    model_load_dir = model_save_dir # where to load model checkpoints from if one is being laoded

    max_length = 64 # max number of tokens in length of tokenized input
    lr = 3e-2 # learning rate
    num_epochs = 1 if debug else 3 # use 1 epoch if debug, else 3 epochs
    batch_size = 12
    test_split_size = 0.1 # 10% for test split

    



    print("\n=== Initialize model ===")

    # Initialize the Llama-2-7b-chat model from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
    model = get_peft_model(model, peft_config) # add PEFT pieces to the LLM
    model.print_trainable_parameters()






    print("\n=== Load dataset ===")

    text_column = "text_input"
    label_column = "text_label"
    
    negatvie_csv_path = "../all-exemplars-pruned/negative.csv"
    positive_csv_path = "../all-exemplars-pruned/positive.csv"

    # load the data from the csvs into a pandas dataframe and group by generic to have only 1 instance of each generic in the dataset
    df_neg = pandas.read_csv(negatvie_csv_path)
    df_neg = df_neg.groupby(['generic']).first()
    df_pos = pandas.read_csv(positive_csv_path)
    df_pos = df_pos.groupby(['generic']).first()

    # load the dataframe into a datset
    dataset_neg = Dataset.from_pandas(df_neg)
    dataset_pos = Dataset.from_pandas(df_pos)

    if debug: # If debugging, only use debug_size% of the dataset
        dataset_neg = dataset_neg.train_test_split(test_size=(1 - (0.01 * debug_size)), seed=10)['train']
        dataset_pos = dataset_pos.train_test_split(test_size=(1 - (0.01 * debug_size)), seed=10)['train']

    # Split dataset into train and test split before doing any data augmentation
    dataset_neg = dataset_neg.train_test_split(test_size=test_split_size, seed=10)
    dataset_pos = dataset_pos.train_test_split(test_size=test_split_size, seed=10)

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

    # Positive samples data augmentation: Add "Some " and "True, some " and "No ", and "False, some "
    dataset_pos = dataset_pos.map(
        lambda x: {
            text_column: ["Some " + i.lower() for i in x["generic"]] + ["No " + i.lower() for i in x["generic"]],
            label_column: ["True, some " + i.lower() for i in x["exemplar"]] + ["False, some " + i.lower() for i in x["exemplar"]]
        },
        batched=True,
        num_proc=1,
        remove_columns=dataset_pos["train"].column_names
    )

    # Combine negative and positive datasets
    dataset_train = concatenate_datasets([dataset_neg["train"], dataset_pos["train"]])
    dataset_test = concatenate_datasets([dataset_neg["test"], dataset_pos["test"]])
    dataset = DatasetDict()
    dataset["train"] = dataset_train
    dataset["test"] = dataset_test

    print(dataset_neg)
    print(dataset_neg["train"][0])
    print(dataset_neg["test"][0])
    print()
    print(dataset_pos)
    print(dataset_pos["train"][0])
    print(dataset_pos["test"][0])
    print()
    print(dataset)
    print(dataset["train"][0])
    print(dataset["test"][0])




    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token="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")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # START: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
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
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        
        # Loop through each example in the batch again to pad the input ids, labels and convert them to PyTorch tensors.
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
    # END: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning




    # Apply preprocess function to dataset using the Accelerator which does DeepSpeed integration
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






    # Either train a new model and save it or load a model that was saved previously
    if train:

        print("\n=== Training Model ===")
        # START: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        # END: based on documentation here: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
        
        # save the model
        print("Saving model to", model_save_dir)
        accelerator.save_state(output_dir=model_save_dir)
    


    else:

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


if __name__ == "__main__":
    main()

