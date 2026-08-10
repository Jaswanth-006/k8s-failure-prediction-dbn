import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.benchmark_metrics import MetricsEvaluator

def main():
    parser = argparse.ArgumentParser(description="Phase 5 Metrics Computation")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to experiment directory (e.g., data/experiments/phase5/cpu/single/pilot_cpu_01)")
    args = parser.parse_args()

    evaluator = MetricsEvaluator(args.experiment_dir)
    if not evaluator.load():
        print(f"Failed to load experiment data from {args.experiment_dir}")
        sys.exit(1)

    result = evaluator.evaluate()

    out_dir = os.path.abspath("data/experiments/phase5/results")
    os.makedirs(out_dir, exist_ok=True)

    exp_id = result.get("experiment_id", "unknown")
    out_file = os.path.join(out_dir, f"{exp_id}_metrics.json")

    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Metrics computation complete. Results saved to {out_file}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
