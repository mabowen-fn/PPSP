import os
from typing import Optional, List, Dict
import openai


class LLMClient:
    def __init__(self, model_name: str, provider: str, api_key_env: str = "NONE",
                 base_url: Optional[str] = None, temperature: float = 0.7):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature

        if provider == "ollama":
            self.client = openai.OpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama"
            )
        elif provider == "openai":
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment: {api_key_env}")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
