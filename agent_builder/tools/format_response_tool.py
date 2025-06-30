import json
from typing import Any

class FormatResponseTool:
    """
    Tool to format LLM responses. Can trim whitespace, add prefixes, or apply markdown formatting. If the response is a dict (parsed JSON), show all relevant fields, including tools, choices, and usage. If the response is an object, show all its attributes.
    """
    def run(self, response: Any) -> str:
        # If response is a string, format as before
        if isinstance(response, str):
            formatted = response.strip()
            return f"Formatted Response:\n```\n{formatted}\n```"
        # If response is a dict (parsed JSON), pretty-print relevant fields
        if isinstance(response, dict):
            output = ["Formatted Response (Full Details):"]
            found = False
            for key in ["id", "model", "usage", "tools", "choices", "content", "additional_kwargs", "response_metadata", "type"]:
                if key in response and response[key]:
                    output.append(f"{key}: {json.dumps(response[key], indent=2)}")
                    found = True
            # Show all tools if present
            if "tools" in response and response["tools"]:
                output.append("Tools:")
                output.append(json.dumps(response["tools"], indent=2))
                found = True
            # Show all choices/messages if present
            if "choices" in response and response["choices"]:
                output.append("Choices:")
                for idx, choice in enumerate(response["choices"]):
                    output.append(f"Choice {idx}: {json.dumps(choice, indent=2)}")
                found = True
            if found:
                return "\n".join(output)
            # If nothing found, fall through to show raw dict
        # If response is an object, show all its attributes
        try:
            if hasattr(response, '__dict__'):
                return f"Formatted Response (Raw Object):\n{json.dumps(vars(response), indent=2, default=str)}"
            elif hasattr(response, '__slots__'):
                slot_dict = {slot: getattr(response, slot) for slot in response.__slots__}
                return f"Formatted Response (Raw Object):\n{json.dumps(slot_dict, indent=2, default=str)}"
        except Exception:
            pass
        # Fallback: pretty-print as JSON or str
        try:
            return f"Formatted Response (Raw):\n{json.dumps(response, indent=2, default=str)}"
        except Exception:
            return f"Formatted Response (Unrecognized type): {str(response)}" 