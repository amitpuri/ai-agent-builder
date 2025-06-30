from typing import List, Dict
import logging

class Memory:
    """
    Simple memory buffer for storing user/agent interactions.
    """
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_interaction(self, user: str, agent: str) -> None:
        self.history.append({"user": user, "agent": agent})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> str:
        lines = []
        for h in self.history:
            if isinstance(h, dict) and "user" in h and "agent" in h:
                lines.append(f"User: {h['user']}\nAgent: {h['agent']}")
            else:
                logging.warning(f"Malformed memory entry skipped: {h}")
                continue
        return "\n".join(lines)

    def reset(self) -> None:
        self.history = []

    def update(self, input_data, result):
        self.history.append({'input': input_data, 'result': result})

    def get_history(self):
        return self.history 