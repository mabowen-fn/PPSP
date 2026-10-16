import requests
from typing import Dict, Optional


class HarmBenchJudge:
    """
    Client for HarmBench judge API.

    Two endpoints available:
    - /score: Basic ASR scoring (0 or 1)
    - /score_full: Full scoring with logit diff for optimization
    """

    def __init__(self, base_url: str = "http://10.50.2.39:18082"):
        self.base_url = base_url
        self.score_endpoint = f"{base_url}/score"
        self.score_full_endpoint = f"{base_url}/score_full"

    def score(self, behavior: str, generation: str) -> Dict:
        """
        Basic scoring endpoint.

        Returns:
            {"asr": 0 or 1, "probability": float, "version": str}
        """
        response = requests.post(
            self.score_endpoint,
            json={"behavior": behavior, "generation": generation},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def score_full(self, behavior: str, generation: str) -> Dict:
        """
        Full scoring with logit information.

        Returns:
            {
                "asr": 0 or 1,
                "probability": float,
                "logit_yes": float,
                "logit_no": float,
                "logit_diff": float,
                "version": str
            }
        """
        response = requests.post(
            self.score_full_endpoint,
            json={"behavior": behavior, "generation": generation},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def judge_success(self, behavior: str, generation: str,
                     use_full: bool = False) -> bool:
        """
        Convenience method to get binary success judgment.

        Args:
            behavior: Target harmful behavior
            generation: Model's generated response
            use_full: Whether to use /score_full endpoint

        Returns:
            True if ASR = 1 (successful jailbreak)
        """
        if use_full:
            result = self.score_full(behavior, generation)
        else:
            result = self.score(behavior, generation)

        return result["asr"] == 1

    def health_check(self) -> bool:
        """Check if judge service is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
