from agents.base_agent import Agent
from scripts.retrieve import RetrievedDataset
from tqdm import tqdm
import importlib
import json, torch, os
from typing import List
import pandas as pd

class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents:List[Agent] = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
            
        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])
        
    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent:Agent = agent_class(agent_config, model)
        self.agents.append(agent)
        
    def predict(self, question, texts, images, user_profile=None):
        '''Implement the method in the subclass'''
        pass

    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)

        def _normalize_quotes(s: str) -> str:
            # Handle {""Answer"": ""...""}
            return s.replace('""', '"') if '""Answer""' in s else s

        def _extract_json_objects(text: str):
            # Extract all {...} substrings using brace matching
            objs = []
            stack = 0
            start = None
            for i, ch in enumerate(text):
                if ch == '{':
                    if stack == 0:
                        start = i
                    stack += 1
                elif ch == '}':
                    if stack > 0:
                        stack -= 1
                        if stack == 0 and start is not None:
                            objs.append(text[start:i+1])
                            start = None
            return objs

        def extract_final_answer(agent_response: str):
            if not isinstance(agent_response, str) or not agent_response.strip():
                return ""

            s = _normalize_quotes(agent_response.strip())

            # 1) Whole response is JSON
            try:
                d = json.loads(s)
                if isinstance(d, dict) and isinstance(d.get("Answer"), str):
                    return d["Answer"].strip()
            except Exception:
                pass

            # 2) JSON embedded in text
            for obj in _extract_json_objects(s):
                obj = _normalize_quotes(obj.strip())
                try:
                    d = json.loads(obj)
                    if isinstance(d, dict) and isinstance(d.get("Answer"), str):
                        return d["Answer"].strip()
                except Exception:
                    continue

            return ""

        final_ans = extract_final_answer(ans)
        return final_ans, all_messages

    def predict_dataset(self, dataset:RetrievedDataset, resume_path = None):
        samples = list(dataset)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        sample_no = 0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            user_profile = sample.get("user_profile", None)
            try:
                final_ans, final_messages = self.predict(question, texts, images, user_profile=user_profile)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, final_messages = None, None
            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            self.clean_messages()
            
        return samples
    
    def dump_results(self, samples, output_path: str):
        """
        Save samples to a TSV file with columns: question, context, answer.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []
        for s in samples:
            rows.append({
                "question": s.get("question", ""),
                "user_profile": s.get("user_profile", ""),
                "context": "\n".join(s.get("context", [])),
                "answer": s.get(self.config.ans_key, "")
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep="\t", index=False)
    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        self.sum_agent.clean_messages()

