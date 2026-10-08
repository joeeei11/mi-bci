"""Run the fixed left/right IV-2a offline acceptance benchmark."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from mi_bci import Project
from mi_bci.research import TrainingConfig, load_iv2a, train_and_evaluate


def run(dataset: Path, algorithm: str = "FBCSP+LDA") -> dict:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config = TrainingConfig()
    trials, labels, channels, rate = load_iv2a(dataset, config)
    with tempfile.TemporaryDirectory() as directory:
        project = Project.create(Path(directory) / "benchmark", "IV-2a benchmark")
        try:
            model_id = train_and_evaluate(project, trials, labels, channels, rate,
                                          algorithm, config,
                                          source={"kind": "BCI Competition IV-2a", "file": dataset.name,
                                                  "sha256": digest},
                                          subject="iv2a-" + dataset.stem[1:3])
            metrics = next(model["metrics"] for model in project.models() if model["id"] == model_id)
        finally:
            project.close()
    return {"dataset": dataset.name, "sha256": digest, "algorithm": algorithm,
            "protocol": "left/right, artifact-free, seeded stratified 75/25 holdout",
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "test_trials": metrics["test_trials"], "confusion_matrix": metrics["confusion_matrix"],
            "passed_75pct": metrics["accuracy"] >= .75}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--algorithm", choices=["FBCSP+LDA", "EEGNet"], default="FBCSP+LDA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.dataset, args.algorithm)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["passed_75pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
