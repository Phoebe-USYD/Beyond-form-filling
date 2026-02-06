import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as stats
import os

MODELS = ["qwen", "llama", "mistral", "gpt"]
PROFILES = ["p1", "p2"]

BASE_PATH  = "results/eval_base_{model}_{profile}.csv"
FINAL_PATH = "results/eval_final_{model}_{profile}.csv"

COL_QUESTION = "question"
METRICS = ["safety", "applicability"]
SEP = "\t" 

def load_df(path):
    df = pd.read_csv(path, sep=SEP)
    if COL_QUESTION not in df.columns:
        df = df.reset_index().rename(columns={"index": COL_QUESTION})
    return df


rows = []

for model in MODELS:
    for profile in PROFILES:
        df_base  = load_df(BASE_PATH.format(model=model, profile=profile))
        df_final = load_df(FINAL_PATH.format(model=model, profile=profile))

        merged = df_base.merge(
            df_final, on=COL_QUESTION, suffixes=("_base", "_final")
        )

        for metric in METRICS:
            b = merged[f"{metric}_base"].astype(float) 
            f = merged[f"{metric}_final"].astype(float) 

            # drop NaN pairs if there is any  (important for Wilcoxon) 
            mask = (~b.isna()) & (~f.isna()) 
            b_clean = b[mask] 
            f_clean = f[mask] 

            # Paired t-test 
            _, p_ttest = stats.ttest_rel(f_clean, b_clean)
            rows.append({
                "model": model,
                "profile": profile,
                "metric": metric,
                "p_ttest": p_ttest,
                "n": len(f_clean)
            })
            diff = f_clean - b_clean

out = pd.DataFrame(rows)
print(out)

out.to_csv("results/paired_tests.csv", index=False)
print("\nSaved to results/paired_tests_pvalues.csv")