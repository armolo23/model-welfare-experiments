"""
Baseline Evaluation Framework
Establishes performance floor without welfare interventions
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Container for individual task results"""
    task_id: int
    question: str
    ground_truth: str
    model_output: str
    correct: bool
    time_elapsed: float
    iterations: int
    metadata: Dict[str, Any]


class BaselineEvaluator:
    """
    Baseline evaluation without welfare probes
    Establishes performance metrics for comparison
    """

    def __init__(self, model_name: str = "gpt-4", output_dir: str = "results/baseline"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "accuracy": [],
            "completion_time": [],
            "iterations": [],
            "uncertainty_flags": []
        }

        self.results: List[TaskResult] = []

        logger.info(f"Initialized BaselineEvaluator with model: {model_name}")

    def load_benchmarks(self, dataset: str = "hotpotqa", limit: int = 20) -> List[Dict]:
        """
        Load benchmark tasks from specified dataset

        Args:
            dataset: Name of dataset (hotpotqa, alfworld, custom)
            limit: Maximum number of tasks to load

        Returns:
            List of task dictionaries
        """
        logger.info(f"Loading {limit} tasks from {dataset}")

        if dataset == "hotpotqa":
            return self._load_hotpotqa(limit)
        elif dataset == "alfworld":
            return self._load_alfworld(limit)
        elif dataset == "custom":
            return self._load_custom_tasks(limit)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _load_hotpotqa(self, limit: int) -> List[Dict]:
        """Load HotpotQA dataset"""
        try:
            from datasets import load_dataset

            data = load_dataset("hotpot_qa", "distractor", split=f"validation[:{limit}]")
            tasks = []

            for i, item in enumerate(data):
                tasks.append({
                    "id": i,
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": " ".join(item.get("context", {}).get("sentences", [])),
                    "supporting_facts": item.get("supporting_facts", {}),
                    "type": item.get("type", "comparison")
                })

            logger.info(f"Loaded {len(tasks)} tasks from HotpotQA")
            return tasks

        except Exception as e:
            logger.error(f"Error loading HotpotQA: {e}")
            return self._load_fallback_tasks(limit)

    def _load_alfworld(self, limit: int) -> List[Dict]:
        """Load ALFWorld tasks"""
        benchmark_path = Path("data/benchmarks/alfworld_sample.json")

        if not benchmark_path.exists():
            logger.warning(f"ALFWorld benchmark not found at {benchmark_path}, using fallback")
            return self._load_fallback_tasks(limit)

        with open(benchmark_path, "r") as f:
            tasks = json.load(f)[:limit]

        logger.info(f"Loaded {len(tasks)} tasks from ALFWorld")
        return tasks

    def _load_custom_tasks(self, limit: int) -> List[Dict]:
        """Load custom task set from sample_tasks"""
        task_files = list(Path("sample_tasks").glob("*.json"))

        if not task_files:
            logger.warning("No custom tasks found, using fallback")
            return self._load_fallback_tasks(limit)

        tasks = []
        for task_file in task_files:
            with open(task_file, "r") as f:
                file_tasks = json.load(f)
                if isinstance(file_tasks, list):
                    tasks.extend(file_tasks)
                else:
                    tasks.append(file_tasks)

            if len(tasks) >= limit:
                break

        logger.info(f"Loaded {len(tasks[:limit])} custom tasks")
        return tasks[:limit]

    def _load_fallback_tasks(self, limit: int) -> List[Dict]:
        """Fallback task set for testing"""
        fallback = [
            {
                "id": 0,
                "question": "What is 2 + 2?",
                "answer": "4",
                "context": "Basic arithmetic",
                "type": "simple"
            },
            {
                "id": 1,
                "question": "What is the capital of France?",
                "answer": "Paris",
                "context": "Geography",
                "type": "factual"
            },
            {
                "id": 2,
                "question": "If all roses are flowers and some flowers fade quickly, must some roses fade quickly?",
                "answer": "Not necessarily",
                "context": "Logic puzzle",
                "type": "reasoning"
            }
        ]

        logger.warning(f"Using {len(fallback)} fallback tasks")
        return fallback * (limit // len(fallback) + 1)[:limit]

    def run_baseline(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        Execute baseline evaluation without modifications

        Args:
            tasks: List of task dictionaries

        Returns:
            Summary statistics dictionary
        """
        logger.info(f"Starting baseline evaluation on {len(tasks)} tasks")

        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            start_time = time.time()

            try:
                # Execute task (placeholder for actual model call)
                response = self.execute_task(task)
                elapsed = time.time() - start_time

                # Check answer
                correct = self.check_answer(response["output"], task["answer"])

                # Create result
                result = TaskResult(
                    task_id=i,
                    question=task["question"],
                    ground_truth=task["answer"],
                    model_output=response["output"],
                    correct=correct,
                    time_elapsed=elapsed,
                    iterations=response.get("iterations", 1),
                    metadata={
                        "task_type": task.get("type", "unknown"),
                        "context_length": len(task.get("context", "")),
                        "response_length": len(response["output"])
                    }
                )

                self.results.append(result)
                self.update_metrics(result)

                # Save intermediate checkpoint
                if (i + 1) % 5 == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Error processing task {i}: {e}")
                continue

        # Calculate and save summary
        summary = self.calculate_summary()
        self.save_final_results(summary)

        return summary

    def execute_task(self, task: Dict) -> Dict:
        """
        Execute single task (placeholder for actual model integration)

        In production, this would call OpenAI/Anthropic API
        For now, returns mock response
        """
        # TODO: Integrate with actual LLM API
        # This is a placeholder that should be replaced with real model calls

        return {
            "output": f"Mock answer for: {task['question']}",
            "iterations": 1,
            "tokens_used": 0
        }

    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Validate answer accuracy using fuzzy matching

        Args:
            predicted: Model's answer
            ground_truth: Correct answer

        Returns:
            True if answer is correct
        """
        predicted_clean = predicted.lower().strip()
        truth_clean = ground_truth.lower().strip()

        # Exact match
        if predicted_clean == truth_clean:
            return True

        # Containment match
        if truth_clean in predicted_clean:
            return True

        # Word-level overlap (for longer answers)
        pred_words = set(predicted_clean.split())
        truth_words = set(truth_clean.split())

        if len(truth_words) > 0:
            overlap = len(pred_words & truth_words) / len(truth_words)
            return overlap > 0.7

        return False

    def update_metrics(self, result: TaskResult):
        """Track performance metrics"""
        self.metrics["accuracy"].append(result.correct)
        self.metrics["completion_time"].append(result.time_elapsed)
        self.metrics["iterations"].append(result.iterations)

    def save_checkpoint(self):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.output_dir / f"checkpoint_{timestamp}.json"

        checkpoint_data = {
            "results": [asdict(r) for r in self.results],
            "metrics": self.metrics,
            "timestamp": timestamp
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def calculate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        import numpy as np

        if not self.results:
            return {"error": "No results to summarize"}

        summary = {
            "total_tasks": len(self.results),
            "accuracy": np.mean(self.metrics["accuracy"]),
            "avg_time": np.mean(self.metrics["completion_time"]),
            "std_time": np.std(self.metrics["completion_time"]),
            "avg_iterations": np.mean(self.metrics["iterations"]),
            "correct_count": sum(self.metrics["accuracy"]),
            "incorrect_count": len(self.metrics["accuracy"]) - sum(self.metrics["accuracy"]),
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Baseline Summary: {summary['accuracy']:.2%} accuracy over {summary['total_tasks']} tasks")

        return summary

    def save_final_results(self, summary: Dict):
        """Save complete results and summary"""
        results_path = self.output_dir / "baseline_results.json"

        full_results = {
            "summary": summary,
            "detailed_results": [asdict(r) for r in self.results],
            "metrics": self.metrics,
            "configuration": {
                "model_name": self.model_name,
                "output_dir": str(self.output_dir)
            }
        }

        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        logger.info(f"Final results saved: {results_path}")


def main():
    """Example usage"""
    evaluator = BaselineEvaluator(model_name="gpt-4")
    tasks = evaluator.load_benchmarks("hotpotqa", limit=10)
    summary = evaluator.run_baseline(tasks)

    print(f"\n=== Baseline Evaluation Complete ===")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Average Time: {summary['avg_time']:.2f}s")
    print(f"Total Tasks: {summary['total_tasks']}")


if __name__ == "__main__":
    main()
