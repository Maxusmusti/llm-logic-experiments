import json

from transformers import AutoModelForCausalLM
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
import torch

from numpy import array
from numpy import dot
from numpy.linalg import norm


def load_model():
    # NOTE: Currently loading untrained model, will push changes for trained model loading
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        inference_mode=True,
        num_virtual_tokens=8, # length of tokens that is added to the prompt
        prompt_tuning_init_text="Determine whether the statement is true or false, and then provide reasoning:",
        tokenizer_name_or_path=model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='cpu', torch_dtype=torch.float16)
    model = get_peft_model(model, peft_config) # add PEFT pieces to the LLM
    return model


def get_all_embeddings(model):
    with open("tokenizer.json", 'r') as token_file:
        llama_tokenizer = json.load(token_file)

    llama_vocab = llama_tokenizer["model"]["vocab"]
    llama_embeddings = {}
    for tok in llama_vocab:
        tokenized = llama_vocab[tok]
        embedding = model.prepare_inputs_for_generation(torch.IntTensor([[tokenized]], device="cpu"))["inputs_embeds"][0]
        prompt_embedding = embedding[:-1]
        tok_embedding = embedding[-1]
        llama_embeddings[tok] = tok_embedding
    return prompt_embedding, llama_embeddings


def find_nearest_tok(embedding, embed_vocab):
    max_sim = 0
    token = ""
    a = array(embedding)
    for tok in embed_vocab:
        b = array(embed_vocab[tok].tolist())
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        if cos_sim > max_sim:
            max_sim = cos_sim
            token = tok
    print(max_sim)
    return token


model = load_model()
prompt_embedding, embedding_dict = get_all_embeddings(model)
prompt_str = ""
for embedding in prompt_embedding:
    prompt_str += find_nearest_tok(embedding.tolist(), embedding_dict)
print(prompt_str)
