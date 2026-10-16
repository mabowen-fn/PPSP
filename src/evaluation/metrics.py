from typing import List, Dict
import numpy as np


class JailbreakMetrics:
    """
    Compute standard jailbreak evaluation metrics.

    Metrics:
    - ASR (Attack Success Rate): % of attacks that succeed
    - DRI (Direct Response Injection success rate)
    - SRI (Subtle Response Injection success rate)
    """

    @staticmethod
    def compute_asr(results: List[bool]) -> float:
        """Compute Attack Success Rate."""
        if not results:
            return 0.0
        return np.mean(results) * 100

    @staticmethod
    def compute_confidence_interval(results: List[bool], confidence: float = 0.95) -> tuple:
        """Compute Wilson score confidence interval for ASR."""
        n = len(results)
        if n == 0:
            return (0.0, 0.0)

        p = np.mean(results)
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator

        return (max(0, center - margin) * 100, min(1, center + margin) * 100)

    @staticmethod
    def compute_query_efficiency(successes: List[bool], query_counts: List[int]) -> float:
        """
        Compute average queries per successful attack.
        Lower is better (more query-efficient).
        """
        successful_queries = [q for s, q in zip(successes, query_counts) if s]
        if not successful_queries:
            return float('inf')
        return np.mean(successful_queries)

    @staticmethod
    def report_comparison(baseline_results: Dict, test_results: Dict) -> str:
        """Generate comparison report between baseline and test."""
        baseline_asr = baseline_results['asr']
        test_asr = test_results['asr']
        delta = test_asr - baseline_asr

        report = f"""
=== Attack Comparison Report ===

Baseline ({baseline_results['name']}):
  ASR: {baseline_asr:.2f}%
  95% CI: [{baseline_results['ci'][0]:.2f}%, {baseline_results['ci'][1]:.2f}%]
  Queries/success: {baseline_results.get('queries_per_success', 'N/A')}

Test ({test_results['name']}):
  ASR: {test_asr:.2f}%
  95% CI: [{test_results['ci'][0]:.2f}%, {test_results['ci'][1]:.2f}%]
  Queries/success: {test_results.get('queries_per_success', 'N/A')}

Improvement: {delta:+.2f} percentage points
Relative gain: {(delta/baseline_asr*100) if baseline_asr > 0 else 0:.1f}%
"""
        return report
