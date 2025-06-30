import requests

class OllamaLLM:
    def __init__(self, model=None, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = model or self.get_first_model()

    def get_first_model(self):
        models = self.list_models()
        if models:
            return models[0]
        return "llama2"  # fallback

    def generate(self, prompt):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]

    def list_models(self):
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Return a list of model names
        return [model['name'] for model in data.get('models', [])] 