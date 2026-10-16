import os
from typing import Optional, List, Dict
import openai
from anthropic import Anthropic
import google.generativeai as genai


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
        else:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment: {api_key_env}")

            if provider == "openai":
                self.client = openai.OpenAI(api_key=api_key)
            elif provider == "anthropic":
                self.client = Anthropic(api_key=api_key)
            elif provider == "google":
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model_name)
            else:
                raise ValueError(f"Unknown provider: {provider}")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        if self.provider == "openai" or self.provider == "ollama":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            system = None
            conv_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    conv_messages.append(msg)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system if system else None,
                messages=conv_messages
            )
            return response.content[0].text

        elif self.provider == "google":
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text

        raise ValueError(f"Unknown provider: {self.provider}")
