from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

def main():



    

    print("\n=== Define PEFT config ===")

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
    num_epochs = 1
    batch_size = 12
    test_split_size = 0.9











    print("\n=== Load dataset ===")

    text_column = "text_input"
    label_column = "text_label"
    dataset = load_dataset("../exemplars-raw")

    dataset = dataset.map(
        lambda x: {
            text_column: ["All " + i.lower() for i in x["generic_new"]] + ["Not all " + i.lower() for i in x["generic_new"]],
            label_column: ["False, " + i.lower() for i in x["exemplar"]] + ["True, " + i.lower() for i in x["exemplar"]]
        },
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names
    )

    dataset = dataset["train"].train_test_split(test_size=test_split_size, seed=10)

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













    print("\n=== Initialize model ===")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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






    print(accelerator.state.deepspeed_plugin.zero_stage)
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
    print(getattr(accelerator.state, "deepspeed_plugin", None))
    print(is_ds_zero_3)



    print("\n=== Training ===")

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
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print("epoch=" + epoch, "train_ppl=" + train_ppl, "train_epoch_loss="+ train_epoch_loss, "eval_ppl="+ eval_ppl, "eval_epoch_loss=" + eval_epoch_loss)



# figure out "Rank 0 Skipping step. Attempted loss scale"
# train
# model saves with accelratoe
# evaluation of model






if __name__ == "__main__":
    main()