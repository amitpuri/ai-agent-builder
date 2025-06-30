import requests

class OllamaLLM:
    def __init__(self, model=None, base_url="http://localhost:11434", timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        # Health check before proceeding
        try:
            health_response = requests.get(self.base_url, timeout=self.timeout)
            if health_response.status_code != 200:
                raise RuntimeError(f"Ollama health check failed: {health_response.status_code} {health_response.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama health check failed: {e}")
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
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.Timeout:
            return "The request to Ollama timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"An error occurred with Ollama: {e}"

    def list_models(self):
        running_models = set()
        try:
            # Get running models
            url = f"{self.base_url}/api/running"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                running_models = {model['name'] for model in data.get('models', [])}
        except requests.exceptions.RequestException:
            # Endpoint might not exist on older ollama versions, ignore
            pass

        all_models = []
        try:
            # Get all pulled models
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            all_pulled_models = [model['name'] for model in data.get('models', [])]

            for model in all_pulled_models:
                if model in running_models:
                    all_models.append(f"{model} (Running)")
                else:
                    all_models.append(model)
            return all_models
        except Exception as e:
            print(f"Could not list models from Ollama: {e}")
            return [] 