"""
This file computes the total number of tokens in the prompts, along with an upper bound on the number of tokens in the
responses.
"""
import numpy as np
import pandas as pd
from pyprojroot import here
from transformers import GPT2Tokenizer
from prompt_generation import make_k_shot_prompt, make_rationale_prompt
from pyprojroot import here

# global variables
task_description = "Choose the most appropriate paraphrase of the first sentence."
prompt_types = ["basic", "non_explanation", "QUD", "similarity", "contrast"]
gpt_versions = ["curie", "davinci"]
corpus_set = "test"
temp = 0.9
K = 10

gpt_version_codes = {
    "curie": "text-curie-001",
    "davinci": "text-davinci-002"
}

if __name__ == "__main__":

    # read the relevant corpus
    if corpus_set == "dev":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-dev.csv"))
    elif corpus_set == "test":
        df_corpus = pd.read_csv(here(f"data/katz-corpus/katz-corpus-test.csv"))
    else:
        raise ValueError(f"Invalid corpus set: {corpus_set}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    total_tokens = 0
    for index, row in df_corpus.iterrows():

        for prompt_type in prompt_types:
            for gpt_version in gpt_versions:

                # create the relevant prompt
                if prompt_type == "basic":
                    prompt = make_k_shot_prompt(row["prompt"], task_description, k=K)
                elif prompt_type == "non_explanation":
                    prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                                   k=K, step_by_step=False)
                else:
                    prompt = make_rationale_prompt(row["prompt"], task_description, rationale_type=prompt_type,
                                                   k=K, step_by_step=True)

                # computer the number of tokens in the prompt
                n_tokens = len(tokenizer(prompt)["input_ids"])

                total_tokens += n_tokens

    prompt_types[2] = "QUD_v3"

    # computer the upper bound of the number of output tokens
    longest_response = 0
    for gpt_version in gpt_versions:
        for prompt_type in prompt_types:
            df_responses = pd.read_csv(here("data/model-outputs-old/model_responses_corpus={}-set={}-gpt={}-prompt={}-k={}-temp={}-processed.csv"
                                            .format("katz", corpus_set, gpt_version, prompt_type, K, temp))).dropna(axis=0, subset="model_response")

            df_responses["length"] = df_responses["model_response"].apply(lambda x: len(tokenizer(x)["input_ids"]))
            max_length = df_responses["length"].max()
            if max_length > longest_response:
                longest_response = max_length

    print(f"Longest response: {longest_response}")

    output_upper_bound = longest_response * len(prompt_types) * len(gpt_versions) * 151
    print(f"upper bound on output tokens: {output_upper_bound}")
    print(f"total input tokens: {total_tokens}")

    print(f"overall budget: {total_tokens + output_upper_bound}")
    print(f"per model: {(total_tokens + output_upper_bound) / len(gpt_versions)}")

