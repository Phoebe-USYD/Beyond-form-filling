from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        self.personal_enabled = config.personal.enabled
        if not self.personal_enabled:
            config.agents = config.agents[:1]

        super().__init__(config)
    
    def predict(self, question, texts, images,user_profile=None):
        general_agent = self.agents[0]
        general_response, _ = general_agent.predict(
            question, texts=texts, images=images, with_sys_prompt=True
        )
        # print("### Incoming retrieved chunks:", len(texts))
        print("### General Agent: " + general_response)

        # Case 1: personalization disabled - return general agent response as final output
        if not self.personal_enabled:
            return general_response, None

        # Case 2: personalization enabled
        critical_agent = self.agents[1]
        print(user_profile)
        critical_input = ("Question:\n"+ question+ "\n\nUser Profile:\n"+ user_profile)
        critical_response, _ = critical_agent.predict(critical_input, texts=None, images=None, with_sys_prompt=True)
        print("### Critical Agent: " + critical_response)

        all_messages = (
            "General Agent Output:\n"+ general_response
            + "\n\nCritical/Personal Output:\n"+ critical_response)

        final_ans, final_messages = self.sum(all_messages)
        print("### Final Answer: " + final_ans)

        return final_ans, final_messages
