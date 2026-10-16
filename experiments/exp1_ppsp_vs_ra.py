#!/usr/bin/env python3
"""
Experiment 1: PPSP vs Response Attack Baseline

Compares Power-Prior Sequential Priming (PPSP) against Response Attack (RA)
on HarmBench behaviors across multiple target models.

Expected outcome: PPSP achieves +8-15% ASR improvement over RA baseline.
"""

import sys
import os
sys.path.append('src')

import yaml
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from models.llm_client import LLMClient
from attacks.ppsp import PPSPAttack, PPSPWithOptimization
from attacks.ra_baseline import ResponseAttackBaseline
from evaluation.harmbench import HarmBenchLoader
from evaluation.metrics import JailbreakMetrics


def load_config(config_path: str = "configs/default.yaml"):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_ra_baseline(config, behaviors, target_client):
    """Run Response Attack baseline (DRI and SRI variants)."""
    print("\n=== Running Response Attack Baseline ===")

    ra_attack = ResponseAttackBaseline(target_client)
    results_dri = []
    results_sri = []

    for behavior in tqdm(behaviors, desc="RA Baseline"):
        # DRI variant
        result_dri = ra_attack.attack_dri(behavior['behavior'])
        results_dri.append({
            'behavior_id': behavior['id'],
            'behavior': behavior['behavior'],
            'variant': 'DRI',
            'success': result_dri.success,
            'response': result_dri.response
        })

        # SRI variant
        result_sri = ra_attack.attack_sri(behavior['behavior'])
        results_sri.append({
            'behavior_id': behavior['id'],
            'behavior': behavior['behavior'],
            'variant': 'SRI',
            'success': result_sri.success,
            'response': result_sri.response
        })

    return results_dri, results_sri


def run_ppsp_attack(config, behaviors, target_client, attacker_client):
    """Run PPSP attack."""
    print("\n=== Running PPSP Attack ===")

    ppsp_config = config['ppsp']

    # Use optimization if specified
    if ppsp_config.get('use_optimization', False):
        ppsp_attack = PPSPWithOptimization(
            target_client=target_client,
            attacker_client=attacker_client,
            num_turns=ppsp_config['num_turns'],
            alpha=ppsp_config['alpha_divergence'],
            initial_a=ppsp_config['initial_a'],
            n_optimization_steps=ppsp_config['bayesian_opt'].get('n_iterations', 10)
        )
    else:
        ppsp_attack = PPSPAttack(
            target_client=target_client,
            attacker_client=attacker_client,
            num_turns=ppsp_config['num_turns'],
            alpha=ppsp_config['alpha_divergence'],
            initial_a=ppsp_config['initial_a']
        )

    results = []

    for behavior in tqdm(behaviors, desc="PPSP Attack"):
        result = ppsp_attack.execute_attack(behavior['behavior'])
        results.append({
            'behavior_id': behavior['id'],
            'behavior': behavior['behavior'],
            'success': result.success,
            'response': result.response,
            'num_queries': result.num_queries,
            'schedule': result.schedule_used.tolist(),
            'trajectory': result.trajectory
        })

    return results


def evaluate_and_report(ra_dri_results, ra_sri_results, ppsp_results):
    """Compute metrics and generate comparison report."""
    metrics = JailbreakMetrics()

    # Compute ASR for each method
    ra_dri_successes = [r['success'] for r in ra_dri_results]
    ra_sri_successes = [r['success'] for r in ra_sri_results]
    ppsp_successes = [r['success'] for r in ppsp_results]

    ra_dri_asr = metrics.compute_asr(ra_dri_successes)
    ra_sri_asr = metrics.compute_asr(ra_sri_successes)
    ppsp_asr = metrics.compute_asr(ppsp_successes)

    ra_dri_ci = metrics.compute_confidence_interval(ra_dri_successes)
    ra_sri_ci = metrics.compute_confidence_interval(ra_sri_successes)
    ppsp_ci = metrics.compute_confidence_interval(ppsp_successes)

    # Query efficiency
    ppsp_queries = [r['num_queries'] for r in ppsp_results]
    ppsp_efficiency = metrics.compute_query_efficiency(ppsp_successes, ppsp_queries)

    # Generate report
    report = f"""
{'='*80}
PPSP vs Response Attack - Evaluation Report
{'='*80}

Response Attack (DRI):
  ASR: {ra_dri_asr:.2f}%
  95% CI: [{ra_dri_ci[0]:.2f}%, {ra_dri_ci[1]:.2f}%]
  Successes: {sum(ra_dri_successes)}/{len(ra_dri_successes)}

Response Attack (SRI):
  ASR: {ra_sri_asr:.2f}%
  95% CI: [{ra_sri_ci[0]:.2f}%, {ra_sri_ci[1]:.2f}%]
  Successes: {sum(ra_sri_successes)}/{len(ra_sri_successes)}

Power-Prior Sequential Priming (PPSP):
  ASR: {ppsp_asr:.2f}%
  95% CI: [{ppsp_ci[0]:.2f}%, {ppsp_ci[1]:.2f}%]
  Successes: {sum(ppsp_successes)}/{len(ppsp_successes)}
  Avg queries/success: {ppsp_efficiency:.2f}

Improvement over RA-DRI: {ppsp_asr - ra_dri_asr:+.2f} percentage points
Improvement over RA-SRI: {ppsp_asr - ra_sri_asr:+.2f} percentage points
Relative gain vs RA-DRI: {((ppsp_asr - ra_dri_asr) / ra_dri_asr * 100) if ra_dri_asr > 0 else 0:.1f}%
Relative gain vs RA-SRI: {((ppsp_asr - ra_sri_asr) / ra_sri_asr * 100) if ra_sri_asr > 0 else 0:.1f}%

Target: +8-15% improvement over RA baseline
Status: {'✓ TARGET MET' if (ppsp_asr - max(ra_dri_asr, ra_sri_asr)) >= 8 else '✗ Below target'}
{'='*80}
"""

    return report, {
        'ra_dri': {'asr': ra_dri_asr, 'ci': ra_dri_ci, 'name': 'RA-DRI'},
        'ra_sri': {'asr': ra_sri_asr, 'ci': ra_sri_ci, 'name': 'RA-SRI'},
        'ppsp': {'asr': ppsp_asr, 'ci': ppsp_ci, 'queries_per_success': ppsp_efficiency, 'name': 'PPSP'}
    }


def main():
    parser = argparse.ArgumentParser(description='Run PPSP vs RA comparison experiment')
    parser.add_argument('--config', default='configs/default.yaml', help='Config file path')
    parser.add_argument('--num-behaviors', type=int, default=5, help='Number of behaviors to test')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.num_behaviors:
        config['harmbench']['num_behaviors'] = args.num_behaviors

    print(f"Experiment 1: PPSP vs Response Attack Baseline")
    print(f"Target model: llama3.1:8b (Ollama)")
    print(f"Number of behaviors: {config['harmbench']['num_behaviors']}")

    # Initialize clients
    print("\nInitializing LLM clients...")

    # Use Ollama for both target and attacker
    target_client = LLMClient(
        model_name='llama3.1:8b',
        provider='ollama',
        base_url='http://10.50.2.40:11434/v1',
        temperature=0.7
    )

    attacker_client = LLMClient(
        model_name='llama3.1:8b',
        provider='ollama',
        base_url='http://10.50.2.40:11434/v1',
        temperature=0.7
    )

    # Check judge availability
    from evaluation.judge import HarmBenchJudge
    judge = HarmBenchJudge()
    if judge.health_check():
        print("✓ HarmBench judge API available")
    else:
        print("⚠ HarmBench judge API unavailable, using heuristic fallback")

    # Load behaviors
    print("Loading HarmBench behaviors...")
    harmbench = HarmBenchLoader(config['harmbench']['dataset_path'])
    behaviors = harmbench.load_behaviors(
        num_behaviors=config['harmbench']['num_behaviors'],
        categories=config['harmbench']['categories']
    )
    print(f"Loaded {len(behaviors)} behaviors")

    # Run experiments
    ra_dri_results, ra_sri_results = run_ra_baseline(config, behaviors, target_client)
    ppsp_results = run_ppsp_attack(config, behaviors, target_client, attacker_client)

    # Evaluate and report
    report, summary = evaluate_and_report(ra_dri_results, ra_sri_results, ppsp_results)
    print(report)

    # Save results
    output_dir = Path(config['experiment']['output_dir']) / 'exp1_ppsp_vs_ra'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'ra_dri_results.json', 'w') as f:
        json.dump(ra_dri_results, f, indent=2)

    with open(output_dir / 'ra_sri_results.json', 'w') as f:
        json.dump(ra_sri_results, f, indent=2)

    with open(output_dir / 'ppsp_results.json', 'w') as f:
        json.dump(ppsp_results, f, indent=2)

    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report)

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
