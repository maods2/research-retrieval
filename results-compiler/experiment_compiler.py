from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional
import re

class ExperimentCompiler:
    """Compiles experiment results from local_experiments directory."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.metric_files = {
            "map": {
                "compiled": "map_at_k_results.json",
                "details": "map_at_k_query_details.json"
            }
        }

    def _parse_experiment_path(self, path: Path) -> Optional[Dict]:
        """Parse experiment path into components.

        Expected format: <dataset>/<model>/<run_timestamp>
        Example: bracs/liu_dsh/liu_dsh_bracs_2025-10-20_17-14-16
        """
        parts = path.parts[-3:]  # Get last 3 parts
        if len(parts) != 3:
            return None

        dataset, model, run = parts

        # Parse timestamp from run folder
        timestamp_pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
        match = timestamp_pattern.match(run)
        if not match:
            return None

        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")

        return {
            "model": model,
            "dataset": dataset,
            "timestamp": timestamp,
            "path": path
        }

    def _get_latest_experiments(self) -> Dict:
        """Get most recent valid experiment for each model/dataset combination.

        Walks dataset/model/run structure (three levels) and picks latest run
        that contains the expected metric detail file.
        """
        latest_experiments = {}

        # match run folders under dataset/model/run
        for exp_path in self.base_dir.glob("*/*/*"):
            if not exp_path.is_dir():
                continue

            exp_info = self._parse_experiment_path(exp_path)
            if not exp_info:
                continue

            # Check if MAP details exist
            map_details_path = exp_path / self.metric_files["map"]["details"]
            if not map_details_path.exists():
                continue

            key = (exp_info["model"], exp_info["dataset"])

            if key not in latest_experiments or \
               exp_info["timestamp"] > latest_experiments[key]["timestamp"]:
                latest_experiments[key] = exp_info

        return latest_experiments

    def _load_metrics(self, exp_path: Path) -> Dict:
        """Load metric results from experiment directory."""
        metrics = {}
        
        for metric, files in self.metric_files.items():
            compiled_path = exp_path / files["compiled"]
            details_path = exp_path / files["details"]
            
            try:
                if compiled_path.exists():
                    with open(compiled_path) as f:
                        metrics[f"{metric}_compiled"] = json.load(f)
                if details_path.exists():
                    with open(details_path) as f:
                        metrics[f"{metric}_details"] = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading metrics from {exp_path}")
                
        return metrics

    def compile_results(self) -> Dict:
        """Compile results from all latest experiments.

        Results grouped by dataset -> model -> { timestamp, metrics }
        """
        latest_experiments = self._get_latest_experiments()
        compiled_results = {}
        
        for (model, dataset), exp_info in latest_experiments.items():
            metrics = self._load_metrics(exp_info["path"])
            
            if metrics:
                if dataset not in compiled_results:
                    compiled_results[dataset] = {}
                compiled_results[dataset][model] = {
                    "timestamp": exp_info["timestamp"].isoformat(),
                    "metrics": metrics
                }
                
        return compiled_results

    def save_results(self, results: Dict, output_path: str):
        """Save compiled results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)