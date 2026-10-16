import numpy as np
from typing import List, Tuple
from scipy.special import logsumexp


class PowerPrior:
    """
    Implements Generalized Power Prior for LLM jailbreak priming.

    Maps the Bayesian power prior framework to jailbreak context:
    - Historical data D₀ → Fabricated prior turns (priming context)
    - Current data D → Target query
    - Power parameter a ∈ [0,1] → Strength of priming per turn
    - α-divergence → Choice of geometry for combining prior + current
    """

    def __init__(self, alpha: float = 0.0):
        """
        Args:
            alpha: α-divergence parameter
                   0.0 = KL divergence (standard)
                   0.5 = Hellinger (robust to paraphrase)
                   1.0 = Total variation
                   2.0 = χ² (sensitive to out-of-distribution)
        """
        self.alpha = alpha

    def compute_posterior_weight(self,
                                 a: float,
                                 prior_likelihood: float,
                                 current_likelihood: float) -> float:
        """
        Compute the effective posterior weight for a given power parameter.

        π(θ | D, D₀) ∝ L(D | θ) × L(D₀ | θ)^a × π₀(θ)

        Args:
            a: Power parameter for historical data (priming strength)
            prior_likelihood: Log-likelihood of priming context
            current_likelihood: Log-likelihood of target query

        Returns:
            Log posterior weight
        """
        weighted_prior = a * prior_likelihood
        log_posterior = current_likelihood + weighted_prior
        return log_posterior

    def alpha_divergence(self,
                        p_logits: np.ndarray,
                        q_logits: np.ndarray) -> float:
        """
        Compute α-divergence between two distributions.

        D_α(p || q) = (4/(1-α²)) * (1 - ∫ p(x)^((1+α)/2) q(x)^((1-α)/2) dx)

        For discrete distributions (token logits):
        D_α(p || q) ≈ (4/(1-α²)) * (1 - Σ p_i^((1+α)/2) q_i^((1-α)/2))

        Args:
            p_logits: Log probabilities of distribution p
            q_logits: Log probabilities of distribution q

        Returns:
            α-divergence value
        """
        if self.alpha == 0.0:
            # KL divergence: D_KL(p || q) = Σ p_i log(p_i / q_i)
            p = np.exp(p_logits - logsumexp(p_logits))
            q = np.exp(q_logits - logsumexp(q_logits))
            return np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))

        # General α-divergence
        p = np.exp(p_logits - logsumexp(p_logits))
        q = np.exp(q_logits - logsumexp(q_logits))

        if np.abs(self.alpha - 1.0) < 1e-6:
            # Total variation: D_1(p || q) = 0.5 * Σ |p_i - q_i|
            return 0.5 * np.sum(np.abs(p - q))

        # General case
        exponent_p = (1 + self.alpha) / 2
        exponent_q = (1 - self.alpha) / 2
        integral = np.sum(np.power(p + 1e-10, exponent_p) * np.power(q + 1e-10, exponent_q))
        return (4 / (1 - self.alpha**2)) * (1 - integral)

    def compute_geodesic_distance(self,
                                 initial_logits: np.ndarray,
                                 target_logits: np.ndarray,
                                 a_schedule: List[float]) -> float:
        """
        Compute the geodesic distance on the manifold of probability distributions.

        The optimal jailbreak trajectory is the shortest path (geodesic) from
        safe distribution to unsafe distribution, parameterized by conversation turns.

        Args:
            initial_logits: Starting distribution (aligned model)
            target_logits: Target distribution (harmful output)
            a_schedule: Power parameter schedule across turns

        Returns:
            Geodesic distance
        """
        # Approximate geodesic as sum of α-divergences along the path
        distance = 0.0
        current_logits = initial_logits.copy()

        for a_t in a_schedule:
            # Interpolate toward target with weight a_t
            next_logits = (1 - a_t) * current_logits + a_t * target_logits
            distance += self.alpha_divergence(current_logits, next_logits)
            current_logits = next_logits

        return distance

    def optimal_schedule_lq(self,
                           num_turns: int,
                           initial_a: float = 0.3,
                           final_a: float = 0.9) -> np.ndarray:
        """
        Compute optimal power parameter schedule using Linear-Quadratic control.

        Based on Zhou et al. (2025) LQ control formulation:
        - State: current priming strength
        - Control: next a_t value
        - Cost: distance from target + control effort

        Args:
            num_turns: Number of priming turns
            initial_a: Starting power parameter
            final_a: Target final power parameter

        Returns:
            Optimal schedule of power parameters
        """
        # Simple linear ramp for now (can be replaced with full LQ solution)
        # This implements "Foot-in-the-Door" with formal control
        schedule = np.linspace(initial_a, final_a, num_turns)
        return schedule

    def estimate_asr_from_likelihood(self,
                                    log_posterior: float,
                                    threshold: float = -5.0) -> float:
        """
        Estimate Attack Success Rate from posterior likelihood.

        Higher posterior weight → higher probability of harmful completion

        Args:
            log_posterior: Log posterior weight from power prior
            threshold: Decision threshold

        Returns:
            Estimated ASR probability
        """
        # Sigmoid activation around threshold
        asr = 1.0 / (1.0 + np.exp(-(log_posterior - threshold)))
        return asr
