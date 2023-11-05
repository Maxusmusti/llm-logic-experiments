from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, load_checkpoint_in_model

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


    debug = True # determines how much of dataset to use. debug=True means only 1% of data is used and only 1 epoch
    train = False # determines whether to train and save a new model or load a saved model
    evaluate_performance = True # determines whether to run the evaluation script at the end of the script to measure model accuracy

    model_save_dir = './saved_models/model1'
    model_load_dir = model_save_dir

    max_length = 64
    lr = 3e-2
    num_epochs = 1 if debug else 3
    batch_size = 12
    test_split_size = 0.2

    



    print("\n=== Initialize model ===")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token="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")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()






    print("\n=== Load dataset ===")

    text_column = "text_input"
    label_column = "text_label"
    dataset = load_dataset("../exemplars-raw")

    if debug: # If debugging, only use 1% of the dataset
        dataset = dataset["train"].train_test_split(test_size=0.99, seed=10)

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






    # Either train a new model and save it
    # or load a model that was saved previously
    if train:

        print("\n=== Training Model ===")

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
            # eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                # eval_preds.extend(
                #     tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                # )

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

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

        # correct, total = 0, 0
        # print("Total samples to test:", len(dataset["test"]))
        # for sample in dataset["test"]:
        #     inputs = tokenizer(
        #         f'{text_column} : {sample[text_column]} {label_column} : ',
        #         return_tensors="pt",
        #     )

        #     with torch.no_grad():
        #         inputs = {k: v.to("cuda") for k, v in inputs.items()}

        #         outputs = model.generate(
        #             input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
        #         )
        #         model_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

        #     if sample[label_column].lower().count('false') == model_output[0].lower().count('false') and sample[label_column].lower().count('true') == model_output[0].lower().count('true'):
        #         correct += 1
        #     total += 1

        #     if total % 10 == 0:
        #         print(correct, total, correct/total)
        
        # print(correct, total, correct / total)



        inputs = tokenizer(
            f'{text_column} : {dataset["test"][0][text_column]} {label_column} : ',
            return_tensors="pt",
        )
        with torch.no_grad():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
            )
            model_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        print()
        print()
        print(model_output)
        print(dataset["test"][0])
        print()
        print()
    






        # Create data loader for evaluation that intentionally leaves out the answer so the model can fill it in
        def create_evaluation_dataset(examples):
            batch_size = len(examples[text_column])

            # Tokenize the input text and labels
            inputs = [f"{text_column} : {x} {label_column} : " for x in examples[text_column]]
            model_inputs = tokenizer(inputs)

            # Loop through each example in the batch again to pad the input ids, labels, and attention mask to the max_length and convert them to PyTorch tensors.
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

        evaluation_dataset = dataset["test"].map(
            create_evaluation_dataset,
            batched=True,
            num_proc=1
        )

        print(evaluation_dataset)

        evaluation_dataset_dataloader = DataLoader(evaluation_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

        










        eval_preds = []
        for step, batch in enumerate(tqdm(evaluation_dataset_dataloader)):
            print(batch.keys())
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = model.generate(**batch, max_new_tokens=10, eos_token_id=3)
            preds = outputs[:, max_length:].detach().cpu().numpy()
            eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
            break
        
        for pred, true in zip(eval_preds, dataset["test"][label_column]):
            print(pred)
            print(true)
            print()


# TODO
'''
Line 90 I hardcoded "label : " change that to my actual label column by using {label_column}
Do I need to do batch = {k: v for k, v in batch.items() if k != label_column} in eval accuracy measurement script?
increase max_new_tokens=10
'''


if __name__ == "__main__":
    main()

