library(tidyverse)
library(here)

corpus_set = "test"
gpt_versions = c("curie", "davinci")
prompt_types <- c("basic", "QUD")

prompt_names <- c("basic"="No Rationale", "non_explanation"="True Non-Explanation",
                  "subject_predicate"="Subject-Object", "QUD"="QUD", "similarity"="Similarity")
model_names <- c("curie"="Curie", "davinci"="DaVinci")

K = 10
temp = 0.2

df.merged = data.frame()
for (gpt_version in gpt_versions) {
  for (prompt_type in prompt_types) {
    df.results <- read.csv(here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
                 corpus_set, gpt_version, prompt_type, K, temp)))
    df.results <- df.results |>
      select(Type, ID, appropriateness_score) |>
      mutate(
        gpt_version = model_names[[gpt_version]],
        prompt_type = prompt_names[[prompt_type]]
        )
    
    df.merged = rbind(df.merged, df.results)
  }
}

ggplot(
  data = df.merged,
  mapping = aes(x = appropriateness_score)
) +
  facet_grid(gpt_version ~ prompt_type) +
  geom_histogram(bins = 4, fill="lightblue", color="black") +
  geom_vline(aes(xintercept = mean(appropriateness_score)),col='black',size=2) +
  labs(
    title = "Appropriateness Distribution Comparison",
    x = "Appropriateness Score",
    y = "Count"
  ) +
  theme_minimal() +
  theme(aspect.ratio = 3/5)
ggsave(here("figures/appropriateness_scores.pdf"), bg="white", width=7.5, height=4.5)  
