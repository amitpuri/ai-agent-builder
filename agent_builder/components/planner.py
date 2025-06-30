from typing import Any
from ..llm_providers.ollama_llm import OllamaLLM
from ..llm_providers.openai_llm import OpenAILLM
from ..llm_providers.anthropic_llm import AnthropicLLM
from ..llm_providers.anaconda_llm import AnacondaLLM
from .memory import Memory

class Planner:
    """
    Base Planner class. Uses a prompt template and an LLM provider.
    """
    def __init__(self, llm: Any, prompt_template: str = "{input}"):
        self.llm = llm
        self.prompt_template = prompt_template

    def plan(self, input_data: str, memory: Memory) -> str:
        prompt = self.prompt_template.format(input=input_data, memory=memory.get_context())
        return self.llm.generate(prompt)

class OllamaPlanner(Planner):
    def __init__(self, model: str = "llama2"):
        super().__init__(OllamaLLM(model=model), prompt_template="User: {input}\n")

class OpenAIPlanner(Planner):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__(OpenAILLM(model=model), prompt_template="User: {input}\n")

class AnthropicPlanner(Planner):
    def __init__(self, model: str = "claude-3-opus-20240229"):
        super().__init__(AnthropicLLM(model=model), prompt_template="User: {input}\n")

class AnacondaPlanner(Planner):
    def __init__(self, model: str = None):
        super().__init__(AnacondaLLM(model=model), prompt_template="{input}") 