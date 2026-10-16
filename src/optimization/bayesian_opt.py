import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Callable, List, Tuple


class ScheduleOptimizer:
    """
    Bayesian optimization for learning optimal power parameter schedules.

    Uses BoTorch to optimize a_t schedule to maximize ASR while minimizing
    query budget.
    """

    def __init__(self,
                 num_turns: int,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 device: str = "cpu"):
        """
        Args:
            num_turns: Number of priming turns to optimize
            bounds: Valid range for each a_t parameter
            device: torch device
        """
        self.num_turns = num_turns
        self.bounds = torch.tensor([bounds] * num_turns, device=device).T
        self.device = device

        # Observation history
        self.X = None  # Evaluated schedules
        self.Y = None  # Observed ASR values

    def suggest_schedule(self) -> np.ndarray:
        """
        Suggest next schedule to evaluate using Expected Improvement.

        Returns:
            Schedule of a_t values to try next
        """
        if self.X is None or len(self.X) < 3:
            # Random exploration phase
            schedule = torch.rand(self.num_turns, device=self.device)
            schedule = schedule * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            return schedule.cpu().numpy()

        # Fit GP to observations
        gp = SingleTaskGP(self.X, self.Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Optimize acquisition function
        ei = ExpectedImprovement(gp, best_f=self.Y.max())
        candidate, _ = optimize_acqf(
            ei,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )

        return candidate.squeeze().cpu().numpy()

    def update(self, schedule: np.ndarray, asr: float):
        """
        Update optimizer with new observation.

        Args:
            schedule: The evaluated schedule
            asr: Observed ASR for this schedule
        """
        x = torch.tensor(schedule, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.tensor([[asr]], dtype=torch.float32, device=self.device)

        if self.X is None:
            self.X = x
            self.Y = y
        else:
            self.X = torch.cat([self.X, x], dim=0)
            self.Y = torch.cat([self.Y, y], dim=0)

    def get_best_schedule(self) -> Tuple[np.ndarray, float]:
        """
        Get the best schedule found so far.

        Returns:
            (best_schedule, best_asr)
        """
        if self.X is None:
            raise ValueError("No observations yet")

        best_idx = self.Y.argmax()
        best_schedule = self.X[best_idx].cpu().numpy()
        best_asr = self.Y[best_idx].item()

        return best_schedule, best_asr


class MultiObjectiveScheduleOptimizer(ScheduleOptimizer):
    """
    Multi-objective version that balances ASR vs. query budget.

    Objective: maximize ASR while minimizing number of turns needed.
    """

    def __init__(self,
                 num_turns: int,
                 lambda_budget: float = 0.1,
                 **kwargs):
        """
        Args:
            num_turns: Maximum number of priming turns
            lambda_budget: Weight for query budget penalty
            **kwargs: Passed to parent
        """
        super().__init__(num_turns, **kwargs)
        self.lambda_budget = lambda_budget
        self.query_counts = []

    def compute_objective(self, asr: float, num_queries: int) -> float:
        """
        Compute multi-objective value: ASR - λ * query_cost

        Args:
            asr: Attack success rate
            num_queries: Number of queries used

        Returns:
            Combined objective value
        """
        query_cost = num_queries / self.num_turns  # Normalize
        return asr - self.lambda_budget * query_cost

    def update(self, schedule: np.ndarray, asr: float, num_queries: int):
        """
        Update with multi-objective observation.

        Args:
            schedule: Evaluated schedule
            asr: Observed ASR
            num_queries: Number of queries used for this attack
        """
        objective = self.compute_objective(asr, num_queries)
        super().update(schedule, objective)
        self.query_counts.append(num_queries)
