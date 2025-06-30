from openai import OpenAI, APITimeoutError
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAILLM:
    def __init__(self, model=None, api_key=None, base_url=None, timeout=60):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        self.model = model or self.get_first_model()
        self.timeout = timeout

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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except APITimeoutError:
            return "The request to OpenAI timed out. Please try again."
        except Exception as e:
            return f"An error occurred with OpenAI: {e}" 