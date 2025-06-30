import anthropic
from anthropic import APITimeoutError
import os
from dotenv import load_dotenv
from agent_builder.tools.token_counter_tool import TokenCounterTool

load_dotenv()

class AnthropicLLM:
    def __init__(self, model=None, api_key=None, base_url=None, timeout=60):
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout)
        self.model = model or self.get_first_model()
        self.timeout = timeout
        self.token_counter = TokenCounterTool()

    def get_first_model(self):
        models = self.list_models()
        if models:
            return models[0]
        return "claude-3-opus-20240229"  # fallback

    def list_models(self):
        models_env = os.getenv("ANTHROPIC_MODELS")
        if models_env:
            return [m.strip() for m in models_env.split(",") if m.strip()]
        return [
            "claude-opus-4",
            "claude-sonnet-4"
        ]

    def generate(self, prompt):
        if not self.token_counter.run(prompt):
            return "Aborted by user due to token count."
        try:
            print(f"DEBUG: Using model: {self.model}, API key starts with: {self.api_key[:6]}")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except anthropic.APIStatusError as e:
            if e.status_code == 404:
                return f"Model '{self.model}' is not available to your Anthropic account. Please choose another model."
            return f"An error occurred with Anthropic: {e}"
        except APITimeoutError:
            return "The request to Anthropic timed out. Please try again."
        except Exception as e:
            return f"An error occurred with Anthropic: {e}" 