import requests
import os
from dotenv import load_dotenv

load_dotenv()

class AnacondaLLM:
    def __init__(self, model=None, api_key=None, base_url="http://127.0.0.1:8080", timeout=60):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("ANACONDA_API_KEY")
        self.timeout = timeout
        # Health check before proceeding
        health_url = f"{self.base_url}/health"
        try:
            health_response = requests.get(health_url, timeout=self.timeout)
            if health_response.status_code != 200:
                raise RuntimeError(f"Anaconda LLM health check failed: {health_response.status_code} {health_response.text}")
        except Exception as e:
            raise RuntimeError(f"Anaconda LLM health check failed: {e}")
        self.model = model or self.get_first_model()

    def get_first_model(self):
        models = self.list_models()
        if models:
            return models[0]
        return "meta-llama/Llama-3-8b-Instruct"

    def list_models(self):
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            models = []
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    for m in data["data"]:
                        models.append(m.get("id") or m.get("name") or m.get("model") or m.get("path"))
                elif "models" in data and isinstance(data["models"], list):
                    for m in data["models"]:
                        models.append(m.get("id") or m.get("name") or m.get("model") or m.get("path"))
            elif isinstance(data, list):
                for m in data:
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or m.get("model") or m.get("path"))
                    else:
                        models.append(m)
            models = [m for m in models if m]
            return list(dict.fromkeys(models))
        except Exception as e:
            print("Could not list models from Anaconda AI Navigator:", e)
        return []

    def generate(self, prompt):
        url = f"{self.base_url}/completion"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PowerShell/7.2.0",
            "Connection": "close"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "prompt": prompt
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "content" in data:
                return data["content"].strip()
            elif "text" in data:
                return data["text"].strip()
            elif "choices" in data and data["choices"]:
                return data["choices"][0].get("text", "").strip()
            else:
                return str(data)
        except requests.exceptions.Timeout:
            return "The request to Anaconda timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"An error occurred with Anaconda: {e}"