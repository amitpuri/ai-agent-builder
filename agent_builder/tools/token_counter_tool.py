from typing import Any, Callable

class TokenCounterTool:
    """
    Tool to calculate token count, print it, and confirm with the user before proceeding.
    Implements the unified tool interface: run(input_text: str, context: Any = None) -> bool.
    Optionally supports non-interactive mode via the 'interactive' flag in context.
    """
    def __init__(self, tokenizer: Callable[[str], int] = None):
        # Optionally accept a tokenizer function for more accurate token counting
        self.tokenizer = tokenizer or self.default_tokenizer

    def default_tokenizer(self, text: str) -> int:
        # Simple whitespace-based token count; replace with model-specific tokenizer as needed
        return len(text.split())

    def run(self, input_text: str, context: Any = None) -> bool:
        token_count = self.tokenizer(input_text)
        print(f"Token count: {token_count}")
        interactive = True
        if context and isinstance(context, dict):
            interactive = context.get('interactive', True)
        if not interactive:
            return True
        confirm = input("Proceed with LLM call? (y/n): ").strip().lower()
        return confirm == 'y' 