import os
from typing import Optional, List, Dict
import requests
import openai

# Disable proxy for local network requests
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'


class LLMClient:
    def __init__(self, model_name: str, provider: str, api_key_env: str = "NONE",
                 base_url: Optional[str] = None, temperature: float = 0.7):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.base_url = base_url

        if provider == "ollama":
            # Ollama uses native API, not OpenAI-compatible
            self.base_url = base_url or "http://localhost:11434"
        elif provider == "openai":
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment: {api_key_env}")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        if self.provider == "ollama":
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=120,
                    proxies={'http': None, 'https': None}
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Ollama API error: {e}") from e

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")
