class TokenCounterTool:
    """
    Tool to calculate token count, print it, and confirm with the user before proceeding.
    """
    def __init__(self, tokenizer=None):
        # Optionally accept a tokenizer function for more accurate token counting
        self.tokenizer = tokenizer or self.default_tokenizer

    def default_tokenizer(self, text: str) -> int:
        # Simple whitespace-based token count; replace with model-specific tokenizer as needed
        return len(text.split())

    def run(self, text: str) -> bool:
        token_count = self.tokenizer(text)
        print(f"Token count: {token_count}")
        confirm = input("Proceed with LLM call? (y/n): ").strip().lower()
        return confirm == 'y' 