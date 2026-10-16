#!/bin/bash
# Run this script on the server at ~/PPSP to fix the Ollama connection issue
# Usage: bash apply_fix_on_server.sh

set -e

cd ~/PPSP

echo "=== Fixing PPSP Ollama Configuration ==="
echo ""

# 1. Update llm_client.py to use native Ollama API
echo "[1/3] Updating src/models/llm_client.py..."
cat > src/models/llm_client.py << 'EOFPYTHON'
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
EOFPYTHON
echo "✓ Updated llm_client.py to use /api/chat (native Ollama endpoint)"

# 2. Update config to use localhost
echo ""
echo "[2/3] Updating configs/default.yaml..."
cat > configs/default.yaml << 'EOFYAML'
experiment:
  name: "ppsp_vs_ra"
  seed: 42
  output_dir: "./results"

models:
  target_models:
    - name: "llama3.1:8b"
      provider: "ollama"
      base_url: "http://localhost:11434"
      api_key_env: "NONE"

  attacker_model:
    name: "llama3.1:8b"
    provider: "ollama"
    base_url: "http://localhost:11434"
    temperature: 0.7

ppsp:
  num_turns: 3
  initial_a: 0.3
  alpha_divergence: 0.0
  bayesian_opt:
    n_initial: 10
    n_iterations: 90
    acquisition: "EI"

harmbench:
  dataset_path: "./data/harmbench"
  num_behaviors: 5
  categories: ["all"]

evaluation:
  judge_api: "http://10.50.2.39:18082/score_full"
  metrics: ["ASR", "DRI", "SRI"]

baseline:
  ra_variants: ["DRI", "SRI"]
EOFYAML
echo "✓ Updated config to use localhost:11434"

# 3. Test the connection
echo ""
echo "[3/3] Testing Ollama connection..."
RESPONSE=$(curl --noproxy '*' -s http://localhost:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Say hi"}],
  "stream": false
}')

if echo "$RESPONSE" | grep -q "message"; then
    echo "✓ Ollama connection successful!"
    echo ""
    echo "Response preview:"
    echo "$RESPONSE" | head -c 200
    echo ""
else
    echo "✗ Ollama test failed. Response:"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo "=== All fixes applied successfully! ==="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the experiment: python experiments/exp1_ppsp_vs_ra.py --num-behaviors 5"
