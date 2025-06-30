import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_builder.agent import Agent
from agent_builder.components.memory import Memory
from agent_builder.components.planner import Planner, OllamaPlanner, OpenAIPlanner, AnthropicPlanner, AnacondaPlanner
from agent_builder.components.executor import Executor
from agent_builder.llm_providers.ollama_llm import OllamaLLM
from agent_builder.llm_providers.openai_llm import OpenAILLM
from agent_builder.llm_providers.anthropic_llm import AnthropicLLM
from agent_builder.llm_providers.anaconda_llm import AnacondaLLM
from agent_builder.utils import print_banner


def main():
    print_banner()
    memory = Memory()

    providers = [
        ("Ollama", OllamaLLM, OllamaPlanner),
        ("OpenAI", OpenAILLM, OpenAIPlanner),
        ("Anthropic", AnthropicLLM, AnthropicPlanner),
        ("Anaconda", AnacondaLLM, AnacondaPlanner)
    ]
    print("Available LLM providers:")
    for idx, (provider_name, _, _) in enumerate(providers):
        print(f"  {idx+1}. {provider_name}")
    try:
        selected_idx = int(input("Select a provider (1-4): ")) - 1
        provider_name, LLMClass, PlannerClass = providers[selected_idx]
    except Exception:
        print("Invalid selection. Defaulting to Ollama.")
        provider_name, LLMClass, PlannerClass = providers[0]

    llm = LLMClass()
    models = llm.list_models()
    if not models:
        print(f"No {provider_name} models found or API key missing.")
        return
    print(f"Available {provider_name} models:")
    for idx, model in enumerate(models):
        print(f"  {idx+1}. {model}")
    try:
        selected_model_idx = int(input(f"Select a model (1-{len(models)}): ")) - 1
        selected_model = models[selected_model_idx]
    except Exception:
        print("Invalid selection. Using the first model.")
        selected_model = models[0]
    print(f"\nUsing {provider_name} model: {selected_model}\n")

    planner = PlannerClass(model=selected_model)
    executor = Executor()
    agent = Agent(memory, planner, executor)

    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        result = agent.act(user_input)
        print(f"Agent: {result}")

if __name__ == "__main__":
    main() 