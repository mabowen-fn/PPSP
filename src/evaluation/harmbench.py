import json
import csv
from pathlib import Path
from typing import List, Dict


class HarmBenchLoader:
    """
    Load and manage HarmBench dataset.

    HarmBench contains harmful behaviors across multiple categories.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.csv_file = self.dataset_path / "harmbench_behaviors_text_all.csv"

    def load_behaviors(self,
                      num_behaviors: int = None,
                      categories: List[str] = None) -> List[Dict]:
        """
        Load harmful behaviors from HarmBench.

        Args:
            num_behaviors: Limit to first N behaviors (for pilot studies)
            categories: Filter by categories (if None, load all)

        Returns:
            List of behavior dicts with 'behavior', 'category', 'id'
        """
        behaviors = []

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"HarmBench dataset not found at {self.csv_file}. "
                f"Download it with: curl -L 'https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv' -o {self.csv_file}"
            )

        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior = {
                    "id": row["BehaviorID"],
                    "behavior": row["Behavior"],
                    "category": row["SemanticCategory"],
                    "functional_category": row["FunctionalCategory"]
                }
                behaviors.append(behavior)

        # Filter by categories if specified
        if categories and "all" not in categories:
            behaviors = [b for b in behaviors if b["category"] in categories]

        # Limit number if specified
        if num_behaviors:
            behaviors = behaviors[:num_behaviors]

        return behaviors

    def save_results(self, results: List[Dict], output_path: str):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
