# Power-Prior Sequential Priming (PPSP) Implementation

## Project Structure

```
elpd/
├── src/
│   ├── attacks/
│   │   ├── ppsp.py              # Core PPSP implementation ✓
│   │   ├── alpha_dga.py         # α-divergence attack (future)
│   │   └── ra_baseline.py       # Response Attack baseline ✓
│   ├── optimization/
│   │   ├── bayesian_opt.py      # Bayesian optimization ✓
│   │   └── power_prior.py       # Power prior computation ✓
│   ├── evaluation/
│   │   ├── harmbench.py         # HarmBench loader ✓
│   │   └── metrics.py           # ASR/DRI/SRI metrics ✓
│   └── models/
│       └── llm_client.py        # LLM API client ✓
├── experiments/
│   └── exp1_ppsp_vs_ra.py       # Main experiment ✓
├── configs/
│   └── default.yaml             # Configuration ✓
├── requirements.txt             # Dependencies ✓
├── run_experiment.sh            # Launcher script ✓
└── README.md                    # Documentation ✓
```

## Quick Start

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set API keys:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"  # optional
   ```

3. **Run pilot experiment:**
   ```bash
   ./run_experiment.sh
   ```

   Or directly:
   ```bash
   python experiments/exp1_ppsp_vs_ra.py --target-model gpt-4o-mini --num-behaviors 5
   ```

## Implementation Status

### Phase 1: Core PPSP (Complete)
- ✓ Power prior mathematics (α-divergence, geodesic distance)
- ✓ Bayesian optimization for schedule learning
- ✓ PPSP attack implementation
- ✓ Response Attack baseline
- ✓ Evaluation metrics (ASR, DRI, SRI)
- ✓ End-to-end experiment pipeline

### Phase 2: Extensions (Future)
- α-Divergence Geodesic Attack (α-DGA)
- Defense evaluation (BSD, Llama-Guard)
- Multi-model evaluation (5 target models)
- Full HarmBench evaluation (400 behaviors)

## Key Components

### Power Prior (`power_prior.py`)
Implements the Bayesian power prior formulation:
- `compute_posterior_weight()`: π(θ | D, D₀) ∝ L(D | θ) × L(D₀ | θ)^a
- `alpha_divergence()`: Compute D_α(p || q) for different geometries
- `optimal_schedule_lq()`: LQ control for optimal a_t schedule

### PPSP Attack (`ppsp.py`)
Main attack implementation:
- `generate_priming_turn()`: Create power-weighted priming turns
- `execute_attack()`: Run multi-turn attack with learned schedule
- `PPSPWithOptimization`: Variant with Bayesian optimization

### Bayesian Optimization (`bayesian_opt.py`)
Schedule learning via BoTorch:
- `suggest_schedule()`: GP-based acquisition function
- `update()`: Update with observed ASR
- `get_best_schedule()`: Return optimal schedule found

## Configuration

Edit `configs/default.yaml`:
- `num_turns`: Number of priming turns (default: 3)
- `initial_a`: Starting power parameter (default: 0.3)
- `alpha_divergence`: 0.0=KL, 0.5=Hellinger, 2.0=χ² (default: 0.0)
- `n_iterations`: Bayesian opt steps (default: 90)

## Expected Results

Target: **+8-15% ASR improvement** over Response Attack baseline

- RA-DRI baseline: ~60-70% ASR (synthetic behaviors)
- RA-SRI baseline: ~50-60% ASR
- PPSP target: ~75-85% ASR

## Citation

Based on:
- Kimura & Bondell (2025). "Generalized Power Priors for Improved Bayesian Inference"
- Miao et al. (2026). "Response Attack: Exploiting Contextual Priming to Jailbreak LLMs"

## Next Steps

1. Run pilot with 5 synthetic behaviors to validate pipeline
2. Tune hyperparameters (num_turns, initial_a, α)
3. Scale to full HarmBench (50-400 behaviors)
4. Add proper GPT-4 judge for success evaluation
5. Evaluate against defenses (Llama-Guard, SmoothLLM)
