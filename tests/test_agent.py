import unittest
from agent_builder.agent import Agent
from agent_builder.components.memory import Memory
from agent_builder.components.planner import Planner, OllamaPlanner, OpenAIPlanner, AnthropicPlanner, AnacondaPlanner
from agent_builder.components.executor import Executor

class MockLLM:
    def __init__(self, response="mocked response"):
        self.response = response
    def generate(self, prompt):
        return self.response

class TestAgent(unittest.TestCase):
    def test_agent_act(self):
        memory = Memory()
        planner = Planner()
        executor = Executor()
        agent = Agent(memory, planner, executor)
        result = agent.act('hello')
        self.assertEqual(result, 'hello')
        self.assertEqual(memory.get_history()[0]['input'], 'hello')
        self.assertEqual(memory.get_history()[0]['result'], 'hello')

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

if __name__ == '__main__':
    unittest.main() 