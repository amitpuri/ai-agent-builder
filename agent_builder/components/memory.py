from typing import List, Dict, Any
import logging

class Memory:
    """
    Simple memory buffer for storing user/agent interactions and results.
    Provides unified add_interaction method for all types of memory updates.
    """
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.max_turns = max_turns

    def add_interaction(self, user: str = None, agent: str = None, input_data: Any = None, result: Any = None) -> None:
        try:
            if user is not None and agent is not None:
                self.history.append({"user": user, "agent": agent})
            elif input_data is not None and result is not None:
                self.history.append({'input': input_data, 'result': result})
            else:
                logging.warning(f"Malformed memory entry: user={user}, agent={agent}, input_data={input_data}, result={result}")
                return
            if len(self.history) > self.max_turns:
                self.history.pop(0)
        except Exception as e:
            logging.exception(f"Error adding interaction to memory: {e}")

    def get_context(self) -> str:
        lines = []
        for h in self.history:
            try:
                if isinstance(h, dict) and "user" in h and "agent" in h:
                    lines.append(f"User: {h['user']}\nAgent: {h['agent']}")
                elif isinstance(h, dict) and "input" in h and "result" in h:
                    lines.append(f"Input: {h['input']}\nResult: {h['result']}")
                else:
                    logging.warning(f"Malformed memory entry skipped: {h}")
                    continue
            except Exception as e:
                logging.exception(f"Error processing memory entry: {h}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history 