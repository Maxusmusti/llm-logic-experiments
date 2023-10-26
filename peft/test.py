from transformers import AutoTokenizer, AutoModelForCausalLM

print(1)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

print(2)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

print(3)


