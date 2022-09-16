"""
This file creates an "inverse Katz corpus" which contains literal sentences and options for metaphorical paraphrases
"""
import re
import pandas as pd
from pyprojroot import here
from prompt_generation import make_inverse_katz_prompt

def create_distractors(row, df):
    """
    Create distractor metaphors for a given
    :param row: the row we are creating distractors for
    :param df: the whole dataframe to generate the distractors from
    :return:
    """
    true_metaphor = row["Statement"]
    # determine if the metaphor is singular or plural
    if " are " in true_metaphor:
        plural = True
        verb = " are "
    else:
        plural = False
        verb = " is "

    df["plural"] = df["Statement"].apply(lambda x: True if " are " in x else False)

    # get distractors
    df_number_matched = df[(df["plural"] == plural) & (df["Statement"] != true_metaphor)]
    distractor_sentences = df_number_matched.sample(n=3)["Statement"].values
    stem = true_metaphor.split(verb)[0]
    distractor_ends = [x.split(verb)[1] for x in distractor_sentences]
    distractor_options = [stem + verb + x for x in distractor_ends]

    return distractor_options

train_size = 30
dev_size = 100

if __name__ == "__main__":

    # read in the data
    df_katz = pd.read_csv(here("data/katz-corpus/katz-corpus-full.csv"))

    # exclude the rows we want to exclude
    df_katz = df_katz[df_katz["Include"] == 1]

    inverse_rows = []
    for index, row in df_katz.iterrows():
        distractors = create_distractors(row, df_katz)
        row = {
            "statement": row["Good (4)"],
            "true_answer": row["Statement"]
        }
        for i, distractor in enumerate(distractors):
            row[f"distractor_{i+1}"] = distractor

        inverse_rows.append(row)

    df_inverse_katz = pd.DataFrame(inverse_rows)

    # make the prompts and save the true answer locations
    prompts, indices = [], []
    for index, row in df_inverse_katz.iterrows():
        prompt, index = make_inverse_katz_prompt(row)
        prompts.append(prompt)
        indices.append(index)

    df_inverse_katz["prompt"] = prompts
    df_inverse_katz["index"] = indices

    # split the data into train and dev
    df_inverse_katz = df_inverse_katz.sample(frac=1)
    df_train = df_inverse_katz.iloc[:train_size]
    df_dev = df_inverse_katz.iloc[train_size:train_size+dev_size]
    df_test = df_inverse_katz.iloc[train_size+dev_size:]

    # save the data
    df_train.to_csv(here("data/katz-corpus/inverse-katz-corpus-train.csv"))
    df_dev.to_csv(here("data/katz-corpus/inverse-katz-corpus-dev.csv"))
    df_test.to_csv(here("data/katz-corpus/inverse-katz-corpus-test.csv"))
