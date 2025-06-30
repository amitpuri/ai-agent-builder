import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

models = [
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "claude-opus-4",
    "claude-opus-4-20250514",
    "claude-3-5-haiku",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
]

for model in models:
    print(f"\nTrying model: {model}")
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi"
                        }
                    ]
                }
            ]
        )
        print(f"SUCCESS: {model}")
        print("Response:", message.content)
    except Exception as e:
        print(f"FAILED: {model}")
        print("Error:", e)