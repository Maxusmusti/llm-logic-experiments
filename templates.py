from llama_index.prompts import Prompt, PromptType

# Custom Templates
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Below are generic statements, followed by examples (exemplars) that entail or contradict the statement:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using this information, answer whether the below query is true or false carefully\n"
    "Query: {query_str}\n"
    "Answer: "
)
custom_template = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

BLANK_PROMPT_TMPL = "Query: {query_str}\n" "Answer: "
blank_template = Prompt(BLANK_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)
