library(tidyverse)
library(here)

# constants
corpus_set <- "test"
gpt_version <- "davinci"
K <- 10
temp <- 0.2

# load corpora
df.basic <- read.csv(here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
             corpus_set, gpt_version, "basic", K, temp)))
df.qud <- read.csv(here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
             corpus_set, gpt_version, "QUD", K, temp)))
df.sim <- read.csv(here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
             corpus_set, gpt_version, "similarity", K, temp)))

bot_50_fam <- df.basic |>
  arrange(FAM) |>
  head(50) |>
  select(Type, ID, FAM)
bot_50_cutoff <- max(bot_50_fam$FAM)

top_50_fam <- df.basic |>
  arrange(FAM) |>
  tail(50) |>
  select(Type, ID, FAM)
top_50_cutoff <- min(top_50_fam$FAM)

df.basic <- df.basic |>
  mutate(fam_group = ifelse(FAM >= top_50_cutoff, "top", ifelse(FAM <= bot_50_cutoff, "bottom", "middle")),
         prompt_type = "basic") |>
  select(appropriateness_score, prompt_type, fam_group)
df.qud <- df.qud |>
  mutate(fam_group = ifelse(FAM >= top_50_cutoff, "top", ifelse(FAM <= bot_50_cutoff, "bottom", "middle")),
         prompt_type = "QUD") |>
  select(appropriateness_score, prompt_type, fam_group)
df.sim <- df.sim |>
  mutate(fam_group = ifelse(FAM >= top_50_cutoff, "top", ifelse(FAM <= bot_50_cutoff, "bottom", "middle")),
         prompt_type = "similarity") |>
  select(appropriateness_score, prompt_type, fam_group)

df.all = rbind(df.basic, df.qud, df.sim)

# make the plot
ggplot(
  data = df.all,
  mapping = aes(x = appropriateness_score)
) +
  facet_grid(prompt_type ~ fam_group) + 
  geom_histogram(bins = 4, fill="lightblue", color="black") +
  geom_vline(data=df.all, aes(xintercept = mean(appropriateness_score)),col='red',size=2) +
  labs(
    x = "Appropriateness Score",
    y = "Count",
    title = "Appropriateness score by prompt type and familiarity"
  ) +
  theme_minimal()
