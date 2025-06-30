import unittest
from agent_builder.agent import Agent
from agent_builder.components.memory import Memory
from agent_builder.components.planner import Planner, OllamaPlanner, OpenAIPlanner, AnthropicPlanner, AnacondaPlanner
from agent_builder.components.executor import Executor
from langchain_openai import ChatOpenAI
import sys
from dotenv import load_dotenv
import os

# Load test-specific environment variables if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

class MockLLM:
    def __init__(self, response="mocked response"):
        self.response = response
    def generate(self, prompt):
        return self.response

class LangChainLLMWrapper:
    def __init__(self, model="gpt-3.5-turbo", api_key="sk-test", base_url="https://api.openai.com/v1"):
        self.llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
    def generate(self, prompt):
        # LangChain's invoke returns a Message object, get content
        return self.llm.invoke(prompt).content

class TestAgent(unittest.TestCase):
    def test_agent_act(self):
        memory = Memory()
        planner = Planner(llm=MockLLM("echo"))
        executor = Executor()
        agent = Agent(memory, planner, executor)
        result = agent.act('hello')
        self.assertEqual(result, 'echo')
        self.assertEqual(memory.get_history()[0]['input'], 'hello')
        self.assertEqual(memory.get_history()[0]['result'], 'echo')

    def test_ollama_planner(self):
        planner = OllamaPlanner()
        planner.llm = MockLLM("ollama response")
        agent = Agent(Memory(), planner, Executor())
        result = agent.act('test')
        self.assertEqual(result, 'ollama response')

    def test_openai_planner(self):
        planner = OpenAIPlanner()
        planner.llm = MockLLM("openai response")
        agent = Agent(Memory(), planner, Executor())
        result = agent.act('test')
        self.assertEqual(result, 'openai response')

    def test_anthropic_planner(self):
        planner = AnthropicPlanner()
        planner.llm = MockLLM("anthropic response")
        agent = Agent(Memory(), planner, Executor())
        result = agent.act('test')
        self.assertEqual(result, 'anthropic response')

    def test_anaconda_planner(self):
        planner = AnacondaPlanner()
        planner.llm = MockLLM("anaconda response")
        agent = Agent(Memory(), planner, Executor())
        result = agent.act('test')
        self.assertEqual(result, 'anaconda response')

    def test_langchain_openai_llm(self):
        memory = Memory()
        planner = Planner(llm=MockLLM("langchain response"))
        executor = Executor()
        agent = Agent(memory, planner, executor)
        result = agent.act('hello')
        self.assertEqual(result, 'langchain response')

if __name__ == '__main__':
    print("Select LLM backend:")
    print("  1. LangChain (ChatOpenAI)")
    print("  2. Custom LLM Providers (Ollama, OpenAI, Anthropic, Anaconda)")
    choice = input("Enter choice (1 or 2): ").strip()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if choice == '1':
        suite.addTest(TestAgent('test_langchain_openai_llm'))
    elif choice == '2':
        suite.addTest(TestAgent('test_agent_act'))
        suite.addTest(TestAgent('test_ollama_planner'))
        suite.addTest(TestAgent('test_openai_planner'))
        suite.addTest(TestAgent('test_anthropic_planner'))
        suite.addTest(TestAgent('test_anaconda_planner'))
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    runner = unittest.TextTestRunner()
    runner.run(suite) 