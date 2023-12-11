from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import Dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, load_checkpoint_in_model
import pandas


def main():


    print("\n=== Define PEFT config ===")

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8, # length of tokens that is added to the prompt
        prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
        tokenizer_name_or_path=model_name_or_path,
    )

    model_load_dir = './saved_models/model4' # where to load model checkpoints from if one is being laoded
    max_length = 64
    batch_size = 12



    print("\n=== Initialize model ===")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
    model = get_peft_model(model, peft_config) # add PEFT pieces to the LLM
    model.print_trainable_parameters()



    print("\n=== Load dataset ===")

    text_column = "text_input"
    label_column = "text_label"
    
    csv_path = "demo/data.csv"
    # load the data into a pandas dataframe
    df = pandas.read_csv(csv_path)
    # load the dataframe into a datset
    datasetfrompandas = Dataset.from_pandas(df)
    dataset = DatasetDict()
    dataset["test"] = datasetfrompandas

    print(dataset)
    print(dataset["test"][0])



    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token="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")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply preprocess function to dataset
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    print("\n=== Move model to accelerator to handle device placement ===")

    model = accelerator.prepare(
        model
    )
    accelerator.print(model)


    print("\n=== Loading Model ===")

    # load the model
    print("Loading model from", model_load_dir)
    accelerator.load_state(input_dir=model_load_dir)








        
    print("\n=== Evaluate model ===")

    # Create data loader for evaluation that intentionally leaves out the answer so the model can fill it in
    def create_evaluation_dataset(examples):
        batch_size = len(examples['query'])

        # Define and tokenize the query
        query = [f"{text_column} : {x} Label : " for x in examples['query']]
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

