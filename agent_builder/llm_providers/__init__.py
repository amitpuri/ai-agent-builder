import os
from .anaconda_llm import AnacondaLLM
from .anthropic_llm import AnthropicLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM

class CommandLLM:
    """
    Generic LLM class that selects the provider based on the environment variable LLM_PROVIDER.
    Supported providers: 'anaconda', 'anthropic', 'openai', 'ollama'.
    Reads url, port, and models from environment variables as per each provider's convention.
    """
    def __init__(self, model=None, timeout=60):
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.provider = provider
        self.llm = None
        if provider == "anaconda":
            self.llm = AnacondaLLM(model=model, timeout=timeout)
        elif provider == "anthropic":
            self.llm = AnthropicLLM(model=model, timeout=timeout)
        elif provider == "openai":
            self.llm = OpenAILLM(model=model, timeout=timeout)
        elif provider == "ollama":
            self.llm = OllamaLLM(model=model, timeout=timeout)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    def generate(self, prompt):
        return self.llm.generate(prompt)

    def list_models(self):
        return self.llm.list_models() 