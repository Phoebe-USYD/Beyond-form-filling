from models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class Deepseek7B(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # 8-bit quantization (same idea as your Qwen)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,  # safe for some model repos that define templates/configs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # text-only message format (same as Qwen)
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
        for t in texts or []:
            merged += t + "\n"
        merged += question
        return {"role": "user", "content": merged}

    def _apply_chat_template_fallback(self, messages):
        """
        Some tokenizers may not have a proper chat_template.
        This fallback keeps behavior reasonable for chat-style models.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                parts.append(f"[USER]\n{content}\n")
            else:
                parts.append(f"[ASSISTANT]\n{content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    @torch.no_grad()
    def predict(self, question, texts=None, images=None, history=None):
        self.clean_up()
        messages = self.process_message(question, texts, images, history)

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = self._apply_chat_template_fallback(messages)

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=getattr(self.config, "do_sample", False),
            temperature=getattr(self.config, "temperature", 0.7),
            top_p=getattr(self.config, "top_p", 0.9),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        
        return output_text, messages
