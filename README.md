# AI Agent Builder

A modular Python framework for building, customizing, and running AI agents with pluggable components (memory, planner, executor, etc).

## Features
- Modular agent architecture
- Easy to extend with custom components
- Example agent included
- **Supports multiple LLM providers:**
  - Ollama (local models)
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Anaconda AI Navigator (local or cloud models)
- Provider and model selection at runtime
- `.env` support for API keys

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
│   │   ├── ollama.py
│   │   ├── openai_llm.py
│   │   ├── anthropic_llm.py
│   │   └── anaconda_llm.py
│   └── utils.py
│
├── examples/
│   └── simple_agent.py
│
├── tests/
│   ├── __init__.py
│   └── test_agent.py
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

Copy `.env.example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANACONDA_API_KEY=your-anaconda-api-key-here
```

## Installing as a Package

You can install the project in editable mode (for development):

```bash
pip install -e .
```

This will use the `setup.py` file to install the package and its dependencies, allowing you to import `agent_builder` from anywhere in your environment.

## Building a Distribution

To build a source or wheel distribution:

```bash
python setup.py sdist bdist_wheel
```

## Publishing to PyPI

To publish your package to [PyPI](https://pypi.org/):

1. **Build the distribution:**
   ```bash
   python setup.py sdist bdist_wheel
   ```
2. **Install Twine (if not already):**
   ```bash
   pip install twine
   ```
3. **Upload to PyPI:**
   ```bash
   twine upload dist/*
   ```
   You will be prompted for your PyPI username and password or token.

**Tip:** For test uploads, use [TestPyPI](https://test.pypi.org/):
```bash
twine upload --repository testpypi dist/*
```

For more details, see the [official PyPI packaging guide](https://packaging.python.org/tutorials/packaging-projects/).

## Usage

Run the example agent:

```bash
python examples/simple_agent.py
```

You will be prompted to select an LLM provider (Ollama, OpenAI, Anthropic, Anaconda) and, if applicable, a model.

- **Ollama:** Make sure your Ollama server is running and models are pulled (e.g., `ollama pull llama2`).
- **Anaconda AI Navigator:** Make sure your local server is running and models are available.
- **OpenAI/Anthropic:** Set your API keys in `.env`.

## Testing

```bash
python -m unittest discover tests
```

---

MIT License 