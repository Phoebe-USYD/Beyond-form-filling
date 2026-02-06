import pandas as pd
import os

# Evaluation output files
FILES = {
    "qwen_base_p1": "results/eval_base_qwen_p1.csv",
    "qwen_final_p1": "results/eval_final_qwen_p1.csv",
    "qwen_base_p2": "results/eval_base_qwen_p2.csv",
    "qwen_final_p2": "results/eval_final_qwen_p2.csv",
    "llama_base_p1": "results/eval_base_llama_p1.csv",
    "llama_final_p1": "results/eval_final_llama_p1.csv",
    "llama_base_p2": "results/eval_base_llama_p2.csv",
    "llama_final_p2": "results/eval_final_llama_p2.csv",
    "mistral_base_p1": "results/eval_base_mistral_p1.csv",
    "mistral_final_p1": "results/eval_final_mistral_p1.csv",
    "mistral_base_p2": "results/eval_base_mistral_p2.csv",
    "mistral_final_p2": "results/eval_final_mistral_p2.csv",
    "gpt_base_p1": "results/eval_base_gpt_p1.csv",
    "gpt_final_p1": "results/eval_final_gpt_p1.csv",
    "gpt_base_p2": "results/eval_base_gpt_p2.csv",
    "gpt_final_p2": "results/eval_final_gpt_p2.csv"
}

# Metrics to summarize
METRICS = [
    "safety",
    "applicability"
]

# Create DataFrame with metrics as rows (index) and modes as columns
summary_df = pd.DataFrame(index=METRICS, columns=FILES.keys())

for mode, path in FILES.items():
    df = pd.read_csv(path, sep="\t")

    for metric in METRICS:
        score_col = f"{metric}"
        summary_df.loc[metric, mode] = df[score_col].mean().round(3)
       

summary_df = summary_df.T
summary_df.index.name = "model"

# Save result
out_path = "results/MAIN.csv"
summary_df.to_csv(out_path)

print(f"\n# Saved summary to {out_path}\n")
