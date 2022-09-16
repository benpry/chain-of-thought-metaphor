"""
This file aggregates the model results and creates LaTeX-formatted tables
"""
import pandas as pd
import numpy as np
from pyprojroot import here
from analyze_model_responses import bootstrapped_ci

prompt_types = ["basic", "non_explanation", "subject_predicate", "QUD", "similarity"]

prompt_names = {
    "basic": "No Rationale",
    "non_explanation": "True Non-Explanation",
    "subject_predicate": "Subject-Predicate",
    "QUD": "QUD",
    "similarity": "Similarity"
}

model_types = ["curie", "davinci"]

model_names = {
    "curie": "Curie",
    "davinci": "DaVinci"
}

K = 10
temp = 0.2

if __name__ == "__main__":

    table_str = f"Model "
    all_results = {}

    for prompt_type in prompt_types:
        table_str += f"& {prompt_names[prompt_type]} "
    mean_table_str = table_str[:-1] + "\\\\\\hline\n"
    error_table_str = table_str[:-1] + "\\\\\\hline\n"

    for model_type in model_types:
        mean_table_str += f"{model_names[model_type]} "
        error_table_str += f"{model_names[model_type]} "
        for prompt_type in prompt_types:
            df_responses = pd.read_csv(here(f"data/model-outputs/processed/model_responses_set=test-gpt={model_type}-prompt={prompt_type}-k={K}-temp={temp}-processed.csv"))

            non_parsed_guesses = len(np.where(df_responses["appropriateness_score"].isna())[0])
            mean_score, ci_lower, ci_upper = bootstrapped_ci(df_responses["appropriateness_score"].dropna())

            mean_table_str += "& {:.2f} [{:.2f}, {:.2f}] ".format(mean_score, ci_lower, ci_upper)
            error_table_str += f"& {non_parsed_guesses} "

        mean_table_str += "\\\\\n"
        error_table_str += "\\\\\n"

    print(mean_table_str)
    print(error_table_str)




