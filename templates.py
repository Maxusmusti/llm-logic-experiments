from llama_index.prompts import Prompt, PromptType

# Custom Templates
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Below are generic statements, followed by examples (exemplars) that entail or contradict the statement:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using this information, answer whether the below statement is true or false carefully\n"
    "Statement: {query_str}\n"
    "Answer: "
)
custom_template = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

YN_TEXT_QA_PROMPT_TMPL = (
    "Below are generic statements, followed by examples (exemplars) that entail or contradict the statement:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using this information, answer the below question carefully (yes/no) and explain\n"
    "Question: {query_str}\n"
    "Answer: "
)
yn_template = Prompt(
    YN_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

BLANK_PROMPT_TMPL = "Query: {query_str}\n" "Answer: "
blank_template = Prompt(BLANK_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)
