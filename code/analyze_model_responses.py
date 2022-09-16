"""
This file takes the model outputs and analyzes them
"""
import re

import numpy as np
from pyprojroot import here
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

guess_labels = ["a", "b", "c", "d"]


def extract_guess(response):
    """
    Figure out which response the model chose
    """
    response = response.strip().lower()
    # look for "the answer is X"
    match = re.findall(r"the answer is ([a-d])", response)
    if len(match) == 0:
        # look for "the speaker is saying X"
        match = re.findall(r"the speaker is saying ([a-d])\)", response)
        if len(match) == 0:
            # if there still isn't a match, check for an answer with just the letter
            match = re.findall(r"^([a-d])\)", response)

            if len(match) == 0:
                return None

    return match[0]


def rank_guess(guess, rating_info, task_type):
    """
    Given a guess and list of ratings, get the rating corresponding to the chosen guess
    """
    if task_type == "inverse":
        # rating_info is a single
        return int(rating_info == guess_labels.index(guess) + 1)
    else:
        # rating_info is a list of paraphrase qualities
        return int(rating_info[guess_labels.index(guess)])


def bootstrapped_ci(scores, n=100000):
    """
    Construct a bootstrapped confidence interval
    """
    # do the bootstrapped resampling
    all_bs = np.random.choice(scores, size=(n, len(scores)))
    # compute means within each instance
    means = np.mean(all_bs, axis=1)

    # compute the overall mean and a confidence interval
    mean = np.mean(means)
    ci_lower = np.percentile(means, 2.5)
    ci_upper = np.percentile(means, 97.5)

    # return the mean and confidence interval
    return mean, ci_lower, ci_upper


def random_baseline(ratings, n_questions=100):
    """
    Suppose we randomly selected answers, what ranks would we end up with?
    """
    # choose ratings at random
    rand_ratings = []
    for q in range(n_questions):
        random_rating = int(np.random.choice(ratings))
        rand_ratings.append(random_rating)

    # return all the random ratings
    return rand_ratings

# specify global variables
corpus_set = "test"
gpt_version = "curie"
prompt_type = "similarity"
task_type = "standard"  # "inverse" or "standard"
K = 10
temp = 0.2

if __name__ == "__main__":

    if task_type == "inverse":
        df_responses = pd.read_csv(here(f"data/model-outputs/model_responses_inverse_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.csv"))
    else:
        df_responses = pd.read_csv(here(f"data/model-outputs/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.csv"))

    # drop the religious metaphor that was accidentally included in the test set
    df_responses = df_responses[df_responses["ID"] != 67]

    non_parsed_guesses = 0
    guess_ranks = []
    raw_guesses = []

    print(f"Analyzing {len(df_responses)} responses")

    for index, row in df_responses.iterrows():

        if not isinstance(row["model_response"], str):
            non_parsed_guesses += 1
            guess_ranks.append(None)
            raw_guesses.append(None)
            print("nan response")
            continue

        guess = extract_guess(row["model_response"])
        if guess is None:
            non_parsed_guesses += 1
            guess_ranks.append(None)
            raw_guesses.append(None)
            print(f"couldn't parse guess: {row['model_response']}")
            continue

        if task_type == "inverse":
            rank = rank_guess(guess, int(row["index"]), task_type = task_type)
        else:
            rank = rank_guess(guess, np.fromstring(row["values"][1:-1], sep=" "), task_type = task_type)

        guess_ranks.append(rank)
        raw_guesses.append(guess)

    df_responses["appropriateness_score"] = guess_ranks
    df_responses["raw_guess"] = raw_guesses
    df_responses.to_csv(here(f"data/model-outputs/processed/model_responses_set={corpus_set}-gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}-processed.csv"))

    guess_ranks = [x for x in guess_ranks if x is not None]
    raw_guesses = [x for x in raw_guesses if x is not None]

    mean, ci_lower, ci_upper = bootstrapped_ci(guess_ranks)
    print(f"mean rank: {mean}, [{ci_lower}, {ci_upper}]")

    print(f"{non_parsed_guesses} guesses not parsed")

    random_means = []
    if task_type == "inverse":
        rating_options = [0, 0, 0, 1]
    else:
        rating_options = [1, 2, 3, 4]
    for i in range(10000):
        random_ratings = random_baseline(rating_options, n_questions=len(guess_ranks))
        random_means.append(np.mean(random_ratings))
    random_ci_lower = np.percentile(random_means, 2.5)
    random_ci_upper = np.percentile(random_means, 97.5)
    p_val = len([m for m in random_means if abs(m - 2.5) > abs(mean - 2.5)]) / len(random_means)
    print(f"mean random rating {np.mean(random_means)}, [{random_ci_lower}, {random_ci_upper}]")
    print(f"p-value: {p_val}")

    print(guess_ranks)
    hist = sns.histplot(np.array(guess_ranks), discrete=True)
    hist.set_title(f"Appropriateness Distribution: {gpt_version} with {prompt_type} prompts",
                   fontsize=12)
    if task_type != "inverse":
        hist.set_xticks([1, 2, 3, 4])
    hist.set_xlabel("Appropriateness Score")
    hist.get_figure().savefig(here(f"figures/appropriateness_distribution_gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.png"))

    plt.clf()

    print(raw_guesses)
    hist = sns.countplot(sorted(raw_guesses))
    hist.set_title(f"Response Distribution: {gpt_version} with {prompt_type} prompts",
                   fontsize=12)
    hist.set_xlabel("Response")
    hist.get_figure().savefig(here(f"figures/response_distribution_gpt={gpt_version}-prompt={prompt_type}-k={K}-temp={temp}.png"))
