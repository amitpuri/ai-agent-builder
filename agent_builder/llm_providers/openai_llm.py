from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAILLM:
    def __init__(self, model=None, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or self.get_first_model()

    def get_first_model(self):
        models = self.list_models()
        if models:
            return models[0]
        return "gpt-3.5-turbo"  # fallback

    def list_models(self):
        models = []
        try:
            response = self.client.models.list()
            for m in response.data:
                # Only include chat/completion models
                if "gpt" in m.id or "chat" in m.id:
                    models.append(m.id)
        except Exception as e:
            print("Could not list OpenAI models:", e)
        return models

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip() 