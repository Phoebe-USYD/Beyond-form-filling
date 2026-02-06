import pandas as pd
import os

# Evaluation output files
FILES = {
    "Qwen2-7B-Instruct": "results/eval_qwen.csv",
    "Llama-3.1-8B-Instruct": "results/eval_llama.csv",
    "Mistral-7B-Instruct-v0.3": "results/eval_mistral.csv",
}

# Metrics to summarize
METRICS = [
    "groundedness",
    "answer_relevance",
    "context_relevance",
]

# Create DataFrame with metrics as rows (index) and modes as columns
summary_df = pd.DataFrame(index=METRICS, columns=FILES.keys())

for mode, path in FILES.items():
    df = pd.read_csv(path, sep=",")

    for metric in METRICS:
        score_col = f"{metric}_score"
        summary_df.loc[metric, mode] = df[score_col].mean().round(3)

summary_df = summary_df.T
summary_df.index.name = "model"

# Save result
out_path = "results/RAG_system.csv"
summary_df.to_csv(out_path)

print(f"\n# Saved summary to {out_path}\n")
