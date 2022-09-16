library(tidyverse)
library(tidybayes)
library(brms)
library(here)
library(broom)

create_dfs <- function(df.responses_basic, df.responses_rationale) {
  df.differences <- df.responses_rationale |>
    select(ID, appropriateness_score) |>
    rename(appropriateness_rationale = appropriateness_score) |>
    left_join(df.responses_basic, on=ID) |>
    rename(appropriateness_basic = appropriateness_score) |>
    mutate(appropriateness_diff = appropriateness_rationale - appropriateness_basic) |>
    mutate(appropriateness_rationale_norm = appropriateness_rationale - 2.5)
  
  df.concatenated <- df.responses_rationale |>
    select(ID, appropriateness_score) |>
    mutate(prompt_type = "rationale") |>
    rbind(df.responses_basic |> 
            select(ID, appropriateness_score) |>
            mutate(prompt_type = "basic")
    )
  
  return(list(df.differences, df.concatenated))
}


# constants
corpus_set = "test"
gpt_versions = c("curie", "davinci")
prompt_types = c("basic", "non_explanation", "subject_predicate", "QUD", "similarity")

prompt_names <- c("basic"="No Rationale", "non_explanation"="True Non-Explanation",
               "subject_predicate"="Subject-Object", "QUD"="QUD", "similarity"="Similarity")
model_names <- c("curie"="Curie", "davinci"="DaVinci")

K = 10
temp = 0.2

title_str = "Model "

for (prompt_type in prompt_types) {
  title_str <- sprintf("%s& %s ", title_str, prompt_names[[prompt_type]])
}

cot_str = title_str
fam_str = title_str
thirty_str = title_str


for (gpt_version in gpt_versions) {
  df.responses_basic <- read.csv(
    here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
                 corpus_set, gpt_version, "basic", K, temp)))
  
  cot_str <- sprintf("%s \\\\\n%s ", cot_str, model_names[[gpt_version]])
  fam_str <- sprintf("%s \\\\\n%s ", fam_str, model_names[[gpt_version]])
  thirty_str <- sprintf("%s \\\\\n%s ", thirty_str, model_names[[gpt_version]])
  
    
  for (prompt_type in prompt_types) {
    df.responses_rationale <- read.csv(
      here(sprintf("data/model-outputs/processed/model_responses_set=%s-gpt=%s-prompt=%s-k=%s-temp=%s-processed.csv",
                   corpus_set, gpt_version, prompt_type, K, temp)))
    
    dfs <- create_dfs(df.responses_basic, df.responses_rationale)
    df.differences <- dfs[[1]]
    df.concatenated <- dfs[[2]]
    
    model_cot <- brm(appropriateness_score ~ prompt_type, family=cumulative(), data = df.concatenated)

    cot_summary <- model_cot |>
      as_draws_df() |>
      spread_draws(b_prompt_typerationale) |>
      median_qi()
    
    estimate = round(cot_summary$b_prompt_typerationale[1], digits=2)
    ci_lower = round(cot_summary$.lower[1], digits=2)
    ci_upper = round(cot_summary$.upper[1], digits=2)
    print(sprintf("%s-%s mean improvement with rationales: %s [%s, %s]", gpt_version, prompt_type, estimate, ci_lower, ci_upper))
    cot_str <- sprintf("%s& %s [%s, %s] ", cot_str, estimate, ci_lower, ci_upper)
    
    model_fam <- brm(appropriateness_rationale ~ FAM, family=cumulative(), data=df.differences)
    summary(model_fam)
    
    fam_summary <- model_fam |>
      as_draws_df() |>
      spread_draws(b_FAM) |>
      median_qi()
    
    estimate = round(fam_summary$b_FAM[1], digits=2)
    ci_lower = round(fam_summary$.lower[1], digits=2)
    ci_upper = round(fam_summary$.upper[1], digits=2)
    fam_str <- sprintf("%s& %s [%s, %s] ", fam_str, estimate, ci_lower, ci_upper)
    
    print(sprintf("%s-%s mean familiarity coefficient: %s [%s, %s] ", gpt_version, prompt_type, estimate, ci_lower, ci_upper))
    
    
    top_30_familiar <- slice_max(df.differences, order_by=FAM, n=30)
    bottom_30_familiar <- slice_min(df.differences, order_by=FAM, n=30)
    
    mean_fam_difference = round(mean(top_30_familiar$appropriateness_rationale, na.rm=T) - mean(bottom_30_familiar$appropriateness_rationale, na.rm=T), digits=2)
    thirty_str = sprintf("%s& %s", thirty_str, mean_fam_difference)
    print(sprintf("%s-%s mean familiarity difference: %s", gpt_version,
                  prompt_type,
                  mean_fam_difference))
  }
}

cat(cot_str)
cat(fam_str)
cat(thirty_str)
