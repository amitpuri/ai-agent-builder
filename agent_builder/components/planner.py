from ..llm_providers.ollama_llm import OllamaLLM
from ..llm_providers.openai_llm import OpenAILLM
from ..llm_providers.anthropic_llm import AnthropicLLM
from ..llm_providers.anaconda_llm import AnacondaLLM

class Planner:
    def plan(self, input_data, memory):
        # For demonstration, just echo the input as the plan
        return input_data

class OllamaPlanner:
    def __init__(self, model="llama2"):
        self.llm = OllamaLLM(model=model)

    def plan(self, input_data, memory):
        # Optionally include memory/history in the prompt
        prompt = f"User: {input_data}\n"
        return self.llm.generate(prompt)

class OpenAIPlanner:
    def __init__(self, model="gpt-3.5-turbo"):
        self.llm = OpenAILLM(model=model)

    def plan(self, input_data, memory):
        prompt = f"User: {input_data}\n"
        return self.llm.generate(prompt)

class AnthropicPlanner:
    def __init__(self, model="claude-3-opus-20240229"):
        self.llm = AnthropicLLM(model=model)

    def plan(self, input_data, memory):
        prompt = f"User: {input_data}\n"
        return self.llm.generate(prompt)

class AnacondaPlanner:
    def __init__(self, model=None):
        self.llm = AnacondaLLM(model=model)

    def plan(self, input_data, memory):
        prompt = f"User: {input_data}\n"
        return self.llm.generate(prompt) 