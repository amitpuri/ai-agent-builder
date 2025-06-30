from openai import OpenAI, APITimeoutError
import os
from dotenv import load_dotenv
from agent_builder.tools.token_counter_tool import TokenCounterTool
from agent_builder.tools.format_response_tool import FormatResponseTool

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
        self.token_counter = TokenCounterTool()
        self.formatter = FormatResponseTool()

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
        if not self.token_counter.run(prompt):
            return "Aborted by user due to token count."
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            # Convert OpenAI response to dict for formatting
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            return self.formatter.run(response_dict)
        except APITimeoutError:
            return "The request to OpenAI timed out. Please try again."
        except Exception as e:
            return f"An error occurred with OpenAI: {e}" 