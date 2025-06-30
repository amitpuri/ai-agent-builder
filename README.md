# AI Agent Builder

A modular Python framework for building, customizing, and running AI agents with pluggable components (memory, planner, executor, etc).

## Features
- Modular agent architecture (Memory, Planner, Executor)
- Unified interface for multiple LLM providers
- Easy to extend with custom components or providers
- Example agent included
- **Supports multiple LLM providers:**
  - Ollama (local models)
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Anaconda AI Navigator (local or cloud models)
  - **LangChain-based providers** (OpenAI, Anthropic, Ollama) via `LangChainLLMWrapper`
- Provider and model selection at runtime via environment variables or code
- `.env` support for API keys and configuration
- Robust error handling and health checks
- **Built-in tools for token counting, response formatting, and echoing**

## Project Structure

```
ai-agent-builder/
│
├── agent_builder/
│   ├── __init__.py
│   ├── agent.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── planner.py
│   │   └── executor.py
│   ├── llm_providers/
│   │   ├── __init__.py
│   │   ├── ollama_llm.py
│   │   ├── openai_llm.py
│   │   ├── anthropic_llm.py
│   │   └── anaconda_llm.py
│   ├── tools/
│   │   ├── echo_tool.py
│   │   ├── token_counter_tool.py
│   │   └── format_response_tool.py
│   ├── langchain_llm_wrapper.py
│   └── utils.py
│
├── examples/
│   └── simple_agent.py
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_anaconda_llm.py
│   ├── test_anthropic_llm.py
│   └── test_anthropic_direct.py
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── setup.py
```

## Setup

```bash
cd ai-agent-builder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys and configuration:

```
# Provider selection (default: openai)
LLM_PROVIDER=openai  # or ollama, anthropic, anaconda

# OpenAI
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODELS=claude-3-opus-20240229,claude-opus-4
ANTHROPIC_DEFAULT_MODEL=claude-3-opus-20240229

# Anaconda AI Navigator
ANACONDA_API_KEY=your-anaconda-api-key-here
ANACONDA_BASE_URL=http://127.0.0.1:8080

# Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_DEFAULT_MODEL=llama2
```

## Unified LLM Provider Interface

The `CommandLLM` class provides a unified interface to all supported LLM providers. It selects the provider based on the `LLM_PROVIDER` environment variable (or you can instantiate a provider directly).

```python
from agent_builder.llm_providers import CommandLLM

llm = CommandLLM(model="gpt-3.5-turbo")  # Provider/model can be set via env or argument
response = llm.generate("Hello, world!")
models = llm.list_models()
```

Each provider (`OllamaLLM`, `OpenAILLM`, `AnthropicLLM`, `AnacondaLLM`) implements:
- `generate(prompt)`
- `list_models()`

Providers perform health checks (where applicable) and raise clear errors for missing API keys or connection issues.

## LangChain LLM Wrapper

The `LangChainLLMWrapper` (in `agent_builder/langchain_llm_wrapper.py`) provides a unified interface for LangChain-based providers (OpenAI, Anthropic, Ollama). It supports:
- Token counting and user confirmation before LLM calls (via `TokenCounterTool`)
- Consistent response formatting (via `FormatResponseTool`)
- Easy integration with the agent framework

Example usage:

```python
from agent_builder.langchain_llm_wrapper import LangChainLLMWrapper

llm = LangChainLLMWrapper(provider="openai", model="gpt-4")
response = llm.generate("Your prompt here")
print(response)
```

## Built-in Tools

- **EchoTool** (`agent_builder/tools/echo_tool.py`):
  - Echoes input text. Used for commands like `echo ...`.
- **TokenCounterTool** (`agent_builder/tools/token_counter_tool.py`):
  - Calculates and prints token count for a prompt, asks for user confirmation before LLM call.
- **FormatResponseTool** (`agent_builder/tools/format_response_tool.py`):
  - Formats LLM responses, including all metadata, tool calls, and choices if present. Handles both string and structured (dict/object) responses.

## Modular Agent Components

- **Agent**: Composed of `Memory`, `Planner`, and `Executor`.
- **Memory**: Stores user/agent interactions, supports context retrieval, reset, and update.
- **Planner**: Abstract base class; concrete planners for each LLM provider (`OllamaPlanner`, `OpenAIPlanner`, `AnthropicPlanner`, `AnacondaPlanner`) use provider-specific defaults and prompt templates. The generic `Planner` can be used with any LLM, including LangChainLLMWrapper.
- **Executor**: Handles tool use (e.g., echo), and can be extended for more complex actions.

## Example Usage

Run the example agent:

```bash
python examples/simple_agent.py
```

You will be prompted to select an LLM provider (Ollama, OpenAI, Anthropic, Anaconda, or LangChain-based providers) and, if applicable, a model.

- **Ollama:** Make sure your Ollama server is running and models are pulled (e.g., `ollama pull llama2`).
- **Anaconda AI Navigator:** Make sure your local server is running and models are available.
- **OpenAI/Anthropic:** Set your API keys in `.env`.

## Response Formatting and Tool Usage

- All LLM responses are formatted using `FormatResponseTool`, which displays the main content, metadata, tool calls, and choices (if present).
- When using LangChain or custom providers, token usage is shown and user confirmation is required before each LLM call.
- Echo commands (e.g., `echo "Hello"`) are handled directly by the `Executor` and do not call the LLM.

## Error Handling

- All LLM providers handle timeouts and API errors gracefully, returning user-friendly messages.
- If a required API key is missing, a clear error is raised.
- Health checks are performed on initialization for Anaconda and Ollama providers.

## Extending the Framework

- Add new LLM providers by implementing `generate` and `list_models` methods.
- Extend planners or executors for custom logic or tool use.
- Add new tools in `agent_builder/tools/` and integrate them via the `Executor` or LLM wrappers.

## Testing

```bash
python -m unittest discover tests
```

- Tests cover agent integration, all planners, and memory.
- Mock LLMs are used for isolated testing.
- Direct provider tests for Anaconda and Anthropic (including model listing and prompt generation).
- The test runner in `test_agent.py` allows selection between LangChain and custom providers.

---

MIT License 