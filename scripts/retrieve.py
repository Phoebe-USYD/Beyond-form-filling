import json
import os

class RetrievedDataset:
    def __init__(self, json_path, retriever, top_k=5):
        self.retriever = retriever
        self.top_k = top_k
        self.data = []

        if json_path.endswith(".jsonl"):
            with open(json_path, "r", encoding="utf-8") as f:
                items = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)

        for item in items:
            if isinstance(item, dict):
                question = item["question"]
                user_profile = item.get("user_profile", "")
            else:
                question = item
                user_profile = ""

            self.data.append({
                "question": question,
                "user_profile": user_profile,
            })

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def load_sample_retrieval_data(self, sample):
        question = sample["question"]

        nodes = self.retriever.retrieve(question)
        chunks = [n.get_content() for n in nodes[:self.top_k]]

        sample["context"] = chunks

        images = []  
        return question, chunks, images
