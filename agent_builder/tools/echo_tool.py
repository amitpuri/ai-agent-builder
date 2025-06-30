from typing import Any

class EchoTool:
    """
    A simple tool that echoes the input text.
    Implements the unified tool interface: run(input_text: str, context: Any = None) -> str
    """
    def run(self, input_text: str, context: Any = None) -> str:
        return f"Echo: {input_text}" 