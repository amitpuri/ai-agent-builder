import os
from dotenv import load_dotenv
from agent_builder.llm_providers.anthropic_llm import AnthropicLLM

load_dotenv()

def test_models():
    models_env = os.getenv("ANTHROPIC_MODELS")
    if models_env:
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        # Expanded list with more historical and variant model names
        models = [
            # Claude 4/3/2/1 and variants
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
            "claude-instant-1.1",
            "claude-instant-1.0",
            "claude-1.3",
            "claude-1.2",
            "claude-1.1",
            "claude-1.0",
            # Other possible variants
            "claude-haiku-3.5",
            "claude-haiku-3",
            "claude-opus-3",
            "claude-sonnet-3",
            "claude-sonnet-3.5",
            "claude-haiku-2",
            "claude-opus-2",
            "claude-sonnet-2",
            "claude-haiku-1",
            "claude-opus-1",
            "claude-sonnet-1",
        ]
    prompt = "Say hello in two words"
    print("Testing Anthropic models with prompt:", prompt)
    for model in models:
        print(f"\nTesting model: {model}")
        llm = AnthropicLLM(model=model)
        try:
            result = llm.generate(prompt)
            print("Result:", result)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    test_models() 