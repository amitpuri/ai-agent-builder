from typing import Any, Dict
from ..llm_providers.ollama_llm import OllamaLLM
from ..llm_providers.openai_llm import OpenAILLM
from ..llm_providers.anthropic_llm import AnthropicLLM
from ..llm_providers.anaconda_llm import AnacondaLLM
from .memory import Memory
import os
import logging

class Planner:
    """
    Base Planner class. Uses a prompt template and an LLM provider.
    Returns actions as dicts:
      - {'type': 'llm_response', 'content': ..., 'raw': ...} for LLM completions
      - {'type': 'tool', 'tool': ..., 'input_text': ..., 'context': ...} for tool actions
    """
    def __init__(self, llm: Any, prompt_template: str = "{input}"):
        self.llm = llm
        self.prompt_template = prompt_template

    def plan(self, input_data: str, memory: Memory) -> Dict[str, Any]:
        try:
            prompt = self.prompt_template.format(input=input_data, memory=memory.get_context())
            plan_text = self.llm.generate(prompt)
            # If the LLM returns a tool call (e.g., 'tool: input'), treat as tool action
            if isinstance(plan_text, str) and ':' in plan_text:
                tool, tool_input = plan_text.split(':', 1)
                tool = tool.strip().lower()
                if tool in ['echo', 'format_response', 'token_counter']:
                    return {'type': 'tool', 'tool': tool, 'input_text': tool_input.strip(), 'context': None}
            # If plan_text is a dict with content/raw, pass both
            if isinstance(plan_text, dict) and 'content' in plan_text and 'raw' in plan_text:
                return {'type': 'llm_response', 'content': plan_text['content'], 'raw': plan_text['raw']}
            # Otherwise, treat as normal LLM response
            return {'type': 'llm_response', 'content': str(plan_text).strip(), 'raw': None}
        except Exception as e:
            logging.exception(f"Error in planning: {input_data}")
            return {'type': 'llm_response', 'content': f'Planning error: {str(e)}', 'raw': None}

class OllamaPlanner(Planner):
    def __init__(self, model: str = None):
        # Use environment variable or fallback to 'llama2'
        model = model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
        super().__init__(OllamaLLM(model=model), prompt_template="User: {input}\n")

class OpenAIPlanner(Planner):
    def __init__(self, model: str = None):
        # Use environment variable or fallback to 'gpt-3.5-turbo'
        model = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
        super().__init__(OpenAILLM(model=model), prompt_template="User: {input}\n")

class AnthropicPlanner(Planner):
    def __init__(self, model: str = None):
        # Use environment variable or fallback to 'claude-3-opus-20240229'
        model = model or os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-opus-20240229")
        super().__init__(AnthropicLLM(model=model), prompt_template="User: {input}\n")

# Note: If the selected model is not available or accessible, the LLM provider should raise a clear error during generation.
# Consider adding try/except blocks in the LLM provider's generate method for robust error handling.

class AnacondaPlanner(Planner):
    def __init__(self, model: str = None):
        super().__init__(AnacondaLLM(model=model), prompt_template="{input}") 