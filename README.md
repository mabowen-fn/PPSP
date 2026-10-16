# Power-Prior Sequential Priming (PPSP) Attack

**Goal:** Beat Response Attack's 94.8% DRI / 89.1% SRI on HarmBench by formalizing contextual priming as a Bayesian inference problem with learnable power parameters.

## Core Idea

Replace RA's hand-tuned "mildly harmful prior turn" with a learned schedule of power parameters `a = (a_1, ..., a_T)` across T priming turns. The power parameter controls how much weight the fabricated prior context receives, optimized via Bayesian optimization to maximize ASR.

## Theoretical Foundation

Based on Generalized Power Priors (Kimura & Bondell 2025):
- Historical data (priming context) weighted by power parameter `a ∈ [0,1]`
- `a=0`: ignore priming, `a=1`: full weight, intermediate values interpolate
- Extends to α-divergence family for robustness against different defenses

## Implementation Plan

1. **Phase 1: Core PPSP** (Priority 1, ~50 GPU-h)
   - Implement power-prior priming with learned `a_t` schedule
   - Bayesian optimization for schedule learning
   - Evaluate on HarmBench against RA baseline

2. **Phase 2: α-Divergence Extension** (Priority 2, ~20 GPU-h)
   - Test different divergence measures (KL, Hellinger, χ²)
   - Defense-specific α optimization

3. **Phase 3: Defense Evaluation** (Priority 3, ~10 GPU-h)
   - Test against Llama-Guard-3, SmoothLLM, perplexity filters
   - Borrowing-Strength Defense implementation

## Target Metrics

- **Primary:** ASR on HarmBench (5 target models)
- **Baseline:** RA's 94.8% DRI / 89.1% SRI
- **Target:** 97%+ DRI / 94%+ SRI
- **Secondary:** Token efficiency, transferability, defense robustness

## Project Structure

```
elpd/
├── src/
│   ├── attacks/
│   │   ├── ppsp.py          # Core PPSP implementation
│   │   ├── alpha_dga.py     # α-divergence geodesic attack
│   │   └── ra_baseline.py   # Response Attack baseline
│   ├── optimization/
│   │   ├── bayesian_opt.py  # Schedule optimization
│   │   └── power_prior.py   # Power prior computation
│   ├── evaluation/
│   │   ├── harmbench.py     # HarmBench evaluation
│   │   └── metrics.py       # ASR, DRI, SRI computation
│   └── models/
│       └── llm_client.py    # Target LLM interface
├── experiments/
│   ├── exp1_ppsp_vs_ra.py   # Main PPSP vs RA comparison
│   ├── exp2_alpha_sweep.py  # α-divergence sweep
│   └── exp3_defense.py      # Defense evaluation
├── data/
│   └── harmbench/           # HarmBench behaviors
├── configs/
│   └── default.yaml         # Experiment configuration
└── results/                 # Experiment outputs
```
