from typing import Any, Dict
import logging
from agent_builder.tools.echo_tool import EchoTool
from agent_builder.tools.format_response_tool import FormatResponseTool
from agent_builder.tools.token_counter_tool import TokenCounterTool

class ToolRegistry:
    """
    Registry for tools. Allows registering and retrieving tools by name.
    """
    def __init__(self):
        self.tools = {}

    def register(self, name: str, tool: Any):
        self.tools[name] = tool

    def get(self, name: str):
        return self.tools.get(name)

class Executor:
    """
    Executor for agent actions. Handles:
      - {'type': 'llm_response', 'content': ...}: returns content directly
      - {'type': 'tool', 'tool': ..., 'input_text': ..., 'context': ...}: invokes tool
    """
    def __init__(self, tool_registry: ToolRegistry = None):
        self.tool_registry = tool_registry or ToolRegistry()
        # Register default tools
        self.tool_registry.register('echo', EchoTool())
        self.tool_registry.register('format_response', FormatResponseTool())
        self.tool_registry.register('token_counter', TokenCounterTool())

    def execute(self, action: Dict[str, Any]) -> Any:
        try:
            action_type = action.get('type')
            if action_type == 'llm_response':
                return action.get('content', '')
            elif action_type == 'tool':
                tool_name = action.get('tool')
                input_text = action.get('input_text')
                context = action.get('context', None)
                tool = self.tool_registry.get(tool_name)
                if not tool:
                    logging.error(f"Tool '{tool_name}' not found in registry.")
                    return f"Error: Tool '{tool_name}' not found."
                return tool.run(input_text, context)
            else:
                logging.error(f"Unknown action type: {action_type}")
                return f"Error: Unknown action type '{action_type}'"
        except Exception as e:
            logging.exception(f"Error executing action: {action}")
            return f"Execution error: {str(e)}" 