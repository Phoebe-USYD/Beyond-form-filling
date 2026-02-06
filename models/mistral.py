from models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class Mistral7(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            padding_side="left",
            use_fast=True,
        )

        # Ensure pad token exists (Mistral often uses eos as pad)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # text-only message format
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question,
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans,
        }

    def create_text_message(self, texts, question):
        merged = ""
        if texts:
            for t in texts:
                merged += t + "\n"
        merged += question
        return {"role": "user", "content": merged}

    @torch.no_grad()
    def predict(self, question, texts=None, images=None, history=None):
        self.clean_up()

        messages = self.process_message(question, texts, images, history)

        # Mistral Instruct supports chat template in recent transformers versions.
        # Fallback: simple concat if template missing.
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Minimal fallback formatting
            prompt = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt += f"{role.upper()}: {content}\n"
            prompt += "ASSISTANT:"

        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages
