import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

class AnthropicLLM:
    def __init__(self, model=None, api_key=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model or self.get_first_model()

    def get_first_model(self):
        models = self.list_models()
        if models:
            return models[0]
        return "claude-3-opus-20240229"  # fallback

    def list_models(self):
        # As of now, the Anthropic API does not provide a public endpoint to list models.
        # We'll use a static list of known models, but you can update this as needed.
        # See: https://docs.anthropic.com/claude/docs/models-overview
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]

    def generate(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip() 