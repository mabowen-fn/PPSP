#!/bin/bash
# Quick fix script for server - run this on 10.50.2.40

cd ~/PPSP

echo "Fixing Ollama configuration to use localhost..."

# Fix the config file
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

echo "✓ Config updated to use localhost:11434"
echo ""
echo "Testing Ollama connection..."
curl --noproxy '*' -s http://localhost:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "test"}],
  "stream": false
}' | head -c 100

echo ""
echo ""
echo "If Ollama test worked, run: source .venv/bin/activate && python experiments/exp1_ppsp_vs_ra.py --num-behaviors 5"
