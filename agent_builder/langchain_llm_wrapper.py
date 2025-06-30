from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from agent_builder.tools.token_counter_tool import TokenCounterTool
from agent_builder.tools.format_response_tool import FormatResponseTool

class LangChainLLMWrapper:
    def __init__(self, provider="openai", model=None, api_key=None, base_url=None):
        if provider == "openai":
            self.llm = ChatOpenAI(model=model or "gpt-3.5-turbo", api_key=api_key, base_url=base_url)
        elif provider == "anthropic":
            self.llm = ChatAnthropic(model=model or "claude-3-opus-20240229", api_key=api_key, base_url=base_url)
        elif provider == "ollama":
            self.llm = ChatOllama(model=model or "llama2", base_url=base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self.token_counter = TokenCounterTool()
        self.formatter = FormatResponseTool()
    def generate(self, prompt):
        if not self.token_counter.run(prompt):
            return "Aborted by user due to token count."
        raw_response = self.llm.invoke(prompt)
        # Try to extract metadata if available
        response_dict = {
            "content": getattr(raw_response, "content", str(raw_response)),
            "additional_kwargs": getattr(raw_response, "additional_kwargs", {}),
            "response_metadata": getattr(raw_response, "response_metadata", {}),
            "type": getattr(raw_response, "type", None),
        }
        return self.formatter.run(response_dict) 