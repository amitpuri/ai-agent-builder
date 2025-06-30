from typing import Any
from agent_builder.tools.echo_tool import EchoTool

class Executor:
    """
    Executor for agent actions. For now, just echoes the action, but can be extended for tool use, API calls, etc.
    """
    def __init__(self):
        self.echo_tool = EchoTool()

    def execute(self, action: str, context: Any = None) -> str:
        if action.startswith("echo:"):
            text = action[len("echo:"):].strip()
            return self.echo_tool.run(text)
        return action 