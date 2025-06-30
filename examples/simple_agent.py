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
from agent_builder.llm_providers import CommandLLM
from agent_builder.langchain_llm_wrapper import LangChainLLMWrapper


def get_verbose_choice():
    """Prompt the user to enable or disable verbose output."""
    choice = input("Enable verbose output (show full LLM response details)? (y/n): ").strip().lower()
    return choice == 'y'


def select_from_list(prompt, options):
    """Display a numbered list and return the selected index, or None if invalid."""
    print(prompt)
    for idx, name in enumerate(options, 1):
        print(f"  {idx}. {name}")
    try:
        selection = int(input(f"Select an option (1-{len(options)}): ")) - 1
        if 0 <= selection < len(options):
            return selection
    except Exception:
        pass
    print("Invalid selection.")
    return None


def handle_langchain_backend(verbose, memory):
    """Handle the LangChain backend selection and chat loop."""
    langchain_providers = [
        ("OpenAI", "openai"),
        ("Anthropic", "anthropic"),
        ("Ollama", "ollama")
    ]
    idx = select_from_list("Available LangChain providers:", [n for n, _ in langchain_providers])
    if idx is None:
        return
    provider_name, provider_key = langchain_providers[idx]
    # Get model and API details from environment variables
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
        print("Unsupported provider.")
        return
    llm = LangChainLLMWrapper(provider=provider_key, model=model, api_key=api_key, base_url=base_url)
    planner = Planner(llm=llm, prompt_template="User: {input}\n")
    executor = Executor()
    agent = Agent(memory, planner, executor)
    print(f"Type 'exit' to return to main menu. Using LangChain provider: {provider_name}, model: {model}")
    chat_loop(agent, executor, verbose)


def handle_custom_backend(verbose, memory):
    """Handle the custom backend selection and chat loop for each provider."""
    providers = [
        ("Ollama", OllamaLLM, OllamaPlanner),
        ("OpenAI", OpenAILLM, OpenAIPlanner),
        ("Anthropic", AnthropicLLM, AnthropicPlanner),
        ("Anaconda", AnacondaLLM, AnacondaPlanner)
    ]
    while True:
        options = [p[0] for p in providers] + ["Test all providers (CommandLLM)", "Back to main menu"]
        idx = select_from_list("Available LLM providers:", options)
        if idx is None:
            continue
        if idx == len(providers):
            test_all_providers()
            continue
        if idx == len(providers) + 1:
            break
        provider_name, LLMClass, PlannerClass = providers[idx]
        try:
            llm = LLMClass()
            models = llm.list_models()
            if not models:
                print(f"No {provider_name} models found or API key missing.")
                continue
            model_idx = select_from_list(f"Available {provider_name} models:", models)
            selected_model = models[model_idx] if model_idx is not None else models[0]
            selected_model_name = selected_model if provider_name == "Anaconda" else selected_model.split(" (Running)")[0]
            planner = PlannerClass(model=selected_model_name)
            executor = Executor()
            agent = Agent(memory, planner, executor)
            print("Type 'exit' to return to provider selection.")
            chat_loop(agent, executor, verbose)
        except Exception as e:
            print(f"Error: {e}")


def test_all_providers():
    """Test all providers using CommandLLM and print a sample completion for each."""
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


def chat_loop(agent, executor, verbose):
    """Main chat loop for user interaction with the agent."""
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if user_input.strip().lower().startswith('echo'):
            text = user_input[len('echo'):].strip().strip('"')
            print(f"Agent: {executor.echo_tool.run(text)}")
            continue
        try:
            result = agent.act(user_input, format_response=True, show_full_details=verbose)
            print(f"Agent: {result}")
        except Exception as e:
            print(f"[Error] {type(e).__name__}: {e}")
            print("An error occurred during LLM call. Please check your model name, API key, or provider status.")


def main():
    """Entry point: print banner, prompt for verbosity, and handle backend selection."""
    print_banner()
    memory = Memory()
    verbose = get_verbose_choice()
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
            handle_langchain_backend(verbose, memory)
        elif backend_choice == '2':
            handle_custom_backend(verbose, memory)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 