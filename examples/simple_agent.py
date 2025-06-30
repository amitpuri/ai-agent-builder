import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

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
from agent_builder.llm_providers import CommandLLM
from agent_builder.tools.token_counter_tool import TokenCounterTool
from agent_builder.tools.format_response_tool import FormatResponseTool
from agent_builder.langchain_llm_wrapper import LangChainLLMWrapper


def main():
    print_banner()
    memory = Memory()
    verbose = False

    # Ask user if they want verbose mode
    verbose_choice = input("Enable verbose output (show full LLM response details)? (y/n): ").strip().lower()
    if verbose_choice == 'y':
        verbose = True

    while True:
        print("Select LLM backend:")
        print("  1. LangChain (OpenAI, Anthropic, Ollama)")
        print("  2. Custom LLM Providers (Ollama, OpenAI, Anthropic, Anaconda)")
        print("  3. Exit")
        backend_choice = input("Enter choice (1, 2, or 3): ").strip()

        if backend_choice == '3':
            print("Exiting...")
            break

        if backend_choice == '1':
            # LangChain backend - select provider
            langchain_providers = [
                ("OpenAI", "openai"),
                ("Anthropic", "anthropic"),
                ("Ollama", "ollama")
            ]
            print("Available LangChain providers:")
            for idx, (name, _) in enumerate(langchain_providers):
                print(f"  {idx+1}. {name}")
            try:
                selection = input(f"Select a provider (1-{len(langchain_providers)}): ")
                selected_idx = int(selection) - 1
                provider_name, provider_key = langchain_providers[selected_idx]
            except (ValueError, IndexError):
                print("Invalid selection. Returning to main menu.")
                continue
            # Get model and keys from env or defaults
            if provider_key == "openai":
                model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            elif provider_key == "anthropic":
                model = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-opus-20240229")
                api_key = os.getenv("ANTHROPIC_API_KEY")
                base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            elif provider_key == "ollama":
                model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama2")
                api_key = None
                base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            else:
                print("Unsupported provider. Returning to main menu.")
                continue
            llm = LangChainLLMWrapper(provider=provider_key, model=model, api_key=api_key, base_url=base_url)
            planner = Planner(llm=llm, prompt_template="User: {input}\n")
            executor = Executor()
            agent = Agent(memory, planner, executor)
            print(f"Type 'exit' to return to main menu. Using LangChain provider: {provider_name}, model: {model}")
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                if user_input.strip().lower().startswith('echo'):
                    executor = Executor()
                    text = user_input[len('echo'):].strip().strip('"')
                    print(f"Agent: {executor.echo_tool.run(text)}")
                    continue
                try:
                    # Use verbose output if enabled
                    result = agent.act(user_input, format_response=True, show_full_details=verbose)
                    print(f"Agent: {result}")
                except Exception as e:
                    print(f"[Error] {type(e).__name__}: {e}")
                    print("An error occurred during LLM call. Please check your model name, API key, or provider status.")
            continue

        if backend_choice == '2':
            while True:
                providers = [
                    ("Ollama", OllamaLLM, OllamaPlanner),
                    ("OpenAI", OpenAILLM, OpenAIPlanner),
                    ("Anthropic", AnthropicLLM, AnthropicPlanner),
                    ("Anaconda", AnacondaLLM, AnacondaPlanner)
                ]
                print("Available LLM providers:")
                for idx, (provider_name, _, _) in enumerate(providers):
                    print(f"  {idx+1}. {provider_name}")
                print(f"  {len(providers)+1}. Test all providers (CommandLLM)")
                print(f"  {len(providers)+2}. Back to main menu")
                try:
                    selection = input(f"Select a provider (1-{len(providers)+2}): ")
                    if not selection:
                        print("Invalid selection. Please try again.")
                        continue
                    selected_idx = int(selection) - 1
                    if selected_idx == len(providers):
                        # Separator: Test all providers using CommandLLM
                        test_prompt = "Say hello from {provider}!"
                        test_providers = ["ollama", "openai", "anthropic", "anaconda"]
                        for prov in test_providers:
                            os.environ["LLM_PROVIDER"] = prov
                            print(f"\n--- Testing provider: {prov.upper()} ---")
                            try:
                                llm = CommandLLM()
                                models = llm.list_models()
                                if not models:
                                    print(f"No models found for {prov}.")
                                    continue
                                print(f"First model: {models[0]}")
                                output = llm.generate(test_prompt.format(provider=prov))
                                print(f"Completion: {output}")
                            except Exception as e:
                                print(f"Error with provider {prov}: {e}")
                        continue
                    if selected_idx == len(providers)+1:
                        print("Returning to main menu...")
                        break
                    provider_name, LLMClass, PlannerClass = providers[selected_idx]
                except (ValueError, IndexError):
                    print("Invalid selection. Please try again.")
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

                try:
                    llm = LLMClass()
                    models = llm.list_models()
                    if not models:
                        print(f"No {provider_name} models found or API key missing.")
                        continue
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

                    if provider_name == "Anaconda":
                        selected_model_name = selected_model  # Use full path
                        print(f"[DEBUG] Passing full model path to AnacondaPlanner: {selected_model_name}")
                    else:
                        selected_model_name = selected_model.split(" (Running)")[0]
                    planner = PlannerClass(model=selected_model_name)
                    executor = Executor()
                    agent = Agent(memory, planner, executor)

                    print("Type 'exit' to return to provider selection.")
                    while True:
                        user_input = input("You: ")
                        if user_input.lower() == 'exit':
                            break
                        if user_input.strip().lower().startswith('echo'):
                            executor = Executor()
                            text = user_input[len('echo'):].strip().strip('"')
                            print(f"Agent: {executor.echo_tool.run(text)}")
                            continue
                        try:
                            # Use verbose output if enabled
                            result = agent.act(user_input, format_response=True, show_full_details=verbose)
                            print(f"Agent: {result}")
                        except Exception as e:
                            print(f"[Error] {type(e).__name__}: {e}")
                            print("An error occurred during LLM call. Please check your model name, API key, or provider status.")
                except RuntimeError as e:
                    print(f"\nError: {e}\n")
                    continue


if __name__ == "__main__":
    main() 