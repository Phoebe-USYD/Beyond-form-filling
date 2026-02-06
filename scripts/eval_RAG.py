import os
import pandas as pd
import hydra
import warnings
import logging
from tqdm import tqdm
from trulens.core import Feedback
from trulens_eval import OpenAI
from hydra import compose
warnings.filterwarnings("ignore")
logging.getLogger("trulens_eval").setLevel(logging.ERROR)   

def judge(provider, name): 
    """Create feedback function using TruLens built-in methods with enhanced prompts."""
    def fb_function(input, context, output):

        if name == "groundedness":
            result = provider.groundedness_measure_with_cot_reasons(
                source=context,
                statement=output
            )

        elif name == "answer_relevance":
            result = provider.relevance_with_cot_reasons(
                prompt=input,
                response=output
            )

        elif name == "context_relevance":
            result = provider.context_relevance_with_cot_reasons(
                question=input,
                context=context
            )

        # Extract score and CoT reasoning
        score, reasoning = result
        return score, reasoning
    return Feedback(fb_function, name=name)


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    
    eval_model = cfg.eval.model.name

    model_cfg = compose(config_name=f"model/{eval_model}", overrides=[]).model

    model_name = model_cfg.model_id
    selected_org = cfg.eval.model.selected_org
    api_key = model_cfg.api_key[selected_org]
    os.environ["OPENAI_API_KEY"] = api_key

    provider = OpenAI(model_engine=model_name, api_key=api_key)
    eval_dataset = cfg.eval.input_path
    df = pd.read_csv(eval_dataset, sep="\t")
    records = df.to_dict(orient="records")

    # Build Feedback Metrics
    metrics = [
        "groundedness",
        "answer_relevance",
        "context_relevance",
    ]
    feedbacks = list(judge(provider, m) for m in metrics)

    # Run evaluation
    results = []

    for idx, r in enumerate(tqdm(records, desc="Evaluating")):
        row_results = {
            "question": r["question"],
            "context": r["context"],
            "answer": r["answer"]
        }

        for fb in feedbacks:
            result = fb.imp(r["question"], r["context"], r["answer"])
            if isinstance(result, tuple):
                score, reasoning = result
            else:
                score, reasoning = result, None

            row_results[f"{fb.name}_score"] = score
            row_results[f"{fb.name}_cot"] = reasoning

        results.append(row_results)

    feedback_df = pd.DataFrame(results)

    # Save results
    output_path = cfg.eval.output_path
    feedback_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()