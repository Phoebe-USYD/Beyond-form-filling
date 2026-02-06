from models.base_model import BaseModel
from openai import OpenAI

class GPTOpenAI(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(api_key=self.config.api_key)
        self.model_name = self.config.model_id
        self.temperature = getattr(self.config, "temperature", 0)
        self.max_tokens = getattr(self.config, "max_tokens", 256)

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": question,
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": ans,
        }

    def create_text_message(self, texts, question):
        return {"role": "user", "content": question}

    def predict(self, question, texts=None, images=None, history=None):
        """
        Match BaseModel.predict interface
        Return: (output_text, messages)
        """
        self.clean_up()

        messages = self.process_message(question, texts, images, history)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        output_text = response.choices[0].message.content

        messages.append(self.create_ans_message(output_text))
        self.clean_up()

        return output_text, messages
    