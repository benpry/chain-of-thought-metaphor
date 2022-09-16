"""
This file preprocesses the Katz corpus
"""
import pandas as pd
from pyprojroot import here
from prompt_generation import make_katz_prompt

# determine how big the train and dev sets should be
train_size = 30
dev_size = 100

# train, dev, and test indices (sampled randomly)
train_indices = [83, 138, 35, 55, 239, 262, 112, 81, 273, 245, 72, 135, 143, 0, 268, 175, 267, 39, 98, 305, 265, 110,
                 187, 142, 201, 227, 242, 229, 9, 162]
dev_indices = [217, 251, 168, 127, 22, 23, 290, 189, 154, 294, 295, 123, 118, 172, 19, 255, 246, 284, 28, 221, 103, 1,
               16, 191, 100, 84, 131, 13, 116, 119, 298, 101, 156, 56, 126, 159, 58, 145, 235, 80, 252, 310, 171, 48,
               17, 289, 234, 166, 287, 219, 188, 69, 78, 288, 144, 15, 134, 301, 233, 231, 64, 271, 147, 129, 41, 2,
               249, 130, 91, 272, 203, 243, 44, 293, 86, 75, 139, 195, 302, 150, 125, 185, 182, 181, 57, 21, 68, 312,
               313, 108, 92, 277, 113, 173, 316, 279, 27, 42, 190, 213]
test_indices = [31, 6, 61, 26, 186, 254, 24, 90, 230, 244, 5, 263, 248, 300, 18, 170, 45, 247, 99, 192, 77, 115, 104,
                67, 222, 111, 226, 8, 52, 309, 114, 250, 153, 93, 259, 46, 89, 167, 109, 3, 87, 238, 63, 200, 117, 164,
                10, 205, 47, 282, 151, 59, 65, 155, 122, 37, 314, 216, 165, 210, 36, 38, 270, 132, 82, 163, 152, 177,
                225, 208, 66, 280, 174, 261, 49, 157, 202, 85, 133, 223, 128, 276, 97, 51, 178, 79, 107, 211, 257, 4, 7,
                33, 206, 209, 286, 120, 30, 184, 315, 74, 141, 193, 40, 32, 180, 29, 95, 304, 169, 224, 179, 183, 14,
                12, 240, 281, 194, 269, 278, 232, 136, 71, 303, 140, 62, 307, 137, 285, 283, 291, 160, 158, 292, 20,
                258, 260, 34, 299, 308, 11, 88, 237, 94, 73, 296, 25, 161, 207, 60, 148, 241]

if __name__ == "__main__":

    # read in the data
    df_katz = pd.read_csv(here("data/katz-corpus/katz-corpus-full.csv"))

    # exclude the rows we want to exclude
    df_katz = df_katz[df_katz["Include"] == 1]

    # generate the prompts and values
    prompts_with_values = df_katz.apply(lambda x: make_katz_prompt(x), axis=1)
    prompts = [x[0] for x in prompts_with_values]
    values = [x[1] for x in prompts_with_values]
    df_katz["prompt"], df_katz["values"] = prompts, values

    # create train, dev, and test data
    df_katz_train = df_katz.loc[train_indices]
    df_katz_dev = df_katz.loc[dev_indices]
    df_katz_test = df_katz.loc[test_indices]

    # save the data
    df_katz_train.to_csv(here("data/katz-corpus/katz-corpus-train.csv"), index=False)
    df_katz_dev.to_csv(here("data/katz-corpus/katz-corpus-dev.csv"), index=False)
    df_katz_test.to_csv(here("data/katz-corpus/katz-corpus-test.csv"), index=False)
