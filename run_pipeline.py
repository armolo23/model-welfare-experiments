#!/usr/bin/env python3
"""
Main Pipeline Execution Script
Orchestrates baseline evaluation, welfare-augmented reflexion, and comprehensive analysis
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add module paths
sys.path.append(str(Path(__file__).parent / "baseline_pipeline"))
sys.path.append(str(Path(__file__).parent / "welfare_probe_scripts"))
sys.path.append(str(Path(__file__).parent / "analysis_plots"))

from baseline_eval import BaselineEvaluator
from modified_reflexion import ModifiedReflexionAgent
from evaluation_framework import ComprehensiveEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete experimental pipeline
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline orchestrator - Run ID: {self.run_id}")

    def run_baseline(self) -> Dict:
        """Execute baseline evaluation"""
        logger.info("="*60)
        logger.info("STAGE 1: Baseline Evaluation")
        logger.info("="*60)

        evaluator = BaselineEvaluator(
            model_name=self.config['model'],
            output_dir=str(self.output_dir / "baseline")
        )

        # Load tasks
        tasks = evaluator.load_benchmarks(
            dataset=self.config['dataset'],
            limit=self.config['num_tasks']
        )

        # Run baseline
        baseline_results = evaluator.run_baseline(tasks)

        # Save configuration
        self._save_config(baseline_results, "baseline")

        logger.info(f"Baseline complete - Accuracy: {baseline_results['accuracy']:.2%}")

        return {
            "summary": baseline_results,
            "detailed_results": evaluator.results,
            "metrics": evaluator.metrics
        }

    def run_modified_pipeline(self, tasks: list) -> Dict:
        """Execute modified pipeline with welfare probes"""
        logger.info("="*60)
        logger.info("STAGE 2: Modified Pipeline with Welfare Probes")
        logger.info("="*60)

        agent = ModifiedReflexionAgent(
            model=self.config['model'],
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 500)
        )

        modified_results = {
            "detailed_results": [],
            "welfare_analysis": {}
        }

        total_tasks = len(tasks)

        for i, task in enumerate(tasks):
            logger.info(f"\nProcessing task {i+1}/{total_tasks}")
            logger.info(f"Question: {task['question'][:80]}...")

            try:
                # Run with monitoring
                result = agent.run_with_monitoring(
                    task["question"],
                    max_iterations=self.config.get('max_iterations', 3)
                )

                # Add task metadata
                result["task"] = task
                result["task_id"] = i
                result["correct"] = self._check_answer(
                    result["final_output"],
                    task["answer"]
                )

                modified_results["detailed_results"].append(result)

                logger.info(f"Result: {'✓ Correct' if result['correct'] else '✗ Incorrect'}")
                logger.info(f"Iterations: {result['iterations']}")

            except Exception as e:
                logger.error(f"Error processing task {i}: {e}")
                continue

        # Get aggregated welfare analysis
        modified_results["welfare_analysis"] = agent.welfare_system.analyze_signals()

        # Save results
        result_path = self.output_dir / "modified" / "modified_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(modified_results, f, indent=2, default=str)

        logger.info(f"\nModified pipeline complete")
        logger.info(f"Accuracy: {sum(r['correct'] for r in modified_results['detailed_results']) / len(modified_results['detailed_results']):.2%}")

        return modified_results

    def run_evaluation(self, baseline_results: Dict, modified_results: Dict) -> Dict:
        """Execute comprehensive evaluation"""
        logger.info("="*60)
        logger.info("STAGE 3: Comprehensive Evaluation")
        logger.info("="*60)

        evaluator = ComprehensiveEvaluator(
            output_dir=str(self.output_dir / "evaluation")
        )

        evaluation = evaluator.evaluate_pipeline(baseline_results, modified_results)

        # Save evaluation
        eval_path = self.output_dir / "evaluation" / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2, default=str)

        logger.info("Evaluation complete")

        return evaluation

    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n" + "="*60)
        logger.info("MODEL WELFARE EXPERIMENTS - FULL PIPELINE")
        logger.info("="*60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Model: {self.config['model']}")
        logger.info(f"Dataset: {self.config['dataset']}")
        logger.info(f"Tasks: {self.config['num_tasks']}")
        logger.info("="*60 + "\n")

        try:
            # Stage 1: Baseline
            baseline_results = self.run_baseline()

            # Skip modified pipeline if baseline-only mode
            if self.config.get('baseline_only', False):
                logger.info("\nBaseline-only mode: Skipping modified pipeline")
                self._save_final_summary(baseline_results, None, None)
                return

            # Get tasks for modified pipeline
            evaluator = BaselineEvaluator()
            tasks = evaluator.load_benchmarks(
                dataset=self.config['dataset'],
                limit=self.config['num_tasks']
            )

            # Stage 2: Modified Pipeline
            modified_results = self.run_modified_pipeline(tasks)

            # Stage 3: Evaluation
            evaluation = self.run_evaluation(baseline_results, modified_results)

            # Save complete results
            self._save_final_summary(baseline_results, modified_results, evaluation)

            # Print summary
            self._print_summary(baseline_results, modified_results, evaluation)

            logger.info(f"\n{'='*60}")
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"{'='*60}\n")

        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
            sys.exit(1)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Simple answer checking"""
        predicted_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()

        return truth_lower in predicted_lower or predicted_lower in truth_lower

    def _save_config(self, results: Dict, stage: str):
        """Save configuration for a stage"""
        config_path = self.output_dir / stage / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "run_id": self.run_id,
            "stage": stage,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def _save_final_summary(self,
                           baseline: Dict,
                           modified: Optional[Dict],
                           evaluation: Optional[Dict]):
        """Save complete pipeline summary"""
        summary_path = self.output_dir / "pipeline_summary.json"

        summary = {
            "run_id": self.run_id,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline,
            "modified": modified,
            "evaluation": evaluation
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Pipeline summary saved to: {summary_path}")

    def _print_summary(self, baseline: Dict, modified: Dict, evaluation: Dict):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)

        # Accuracy
        acc_metrics = evaluation.get("accuracy_metrics", {})
        print(f"\nAccuracy:")
        print(f"  Baseline:  {acc_metrics.get('baseline_accuracy', 0):.2%}")
        print(f"  Modified:  {acc_metrics.get('modified_accuracy', 0):.2%}")
        print(f"  Change:    {acc_metrics.get('improvement', 0):+.2%}")

        # Welfare signals
        welfare = evaluation.get("welfare_metrics", {})
        print(f"\nWelfare Signals (Frequency):")
        print(f"  Overload:    {welfare.get('overload_frequency', 0):.2%}")
        print(f"  Ambiguity:   {welfare.get('ambiguity_frequency', 0):.2%}")
        print(f"  Context Req: {welfare.get('context_request_frequency', 0):.2%}")
        print(f"  Confidence:  {welfare.get('confidence_signal_frequency', 0):.2%}")

        # Hallucinations
        hall = evaluation.get("hallucination_metrics", {})
        print(f"\nHallucination Metrics:")
        print(f"  Mean Score: {hall.get('mean_score', 0):.3f}")
        print(f"  High Count: {hall.get('high_hallucination_count', 0)}")

        # Recommendation
        comp = evaluation.get("comparative_analysis", {})
        print(f"\nRecommendation:")
        print(f"  {comp.get('recommendation', 'N/A')}")

        print("\n" + "="*60 + "\n")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Model Welfare Experiments Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline only
  python run_pipeline.py --baseline-only --tasks 10

  # Run full pipeline
  python run_pipeline.py --dataset hotpotqa --tasks 20

  # Custom output directory
  python run_pipeline.py --output-dir experiments/run1 --tasks 15

  # Use specific model
  python run_pipeline.py --model gpt-4 --tasks 20
        """
    )

    parser.add_argument('--baseline-only', action='store_true',
                       help='Run baseline evaluation only')
    parser.add_argument('--dataset', default='hotpotqa',
                       choices=['hotpotqa', 'alfworld', 'custom'],
                       help='Benchmark dataset to use')
    parser.add_argument('--tasks', type=int, default=20,
                       help='Number of tasks to evaluate')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--model', default='gpt-4',
                       help='Model to use (gpt-4, gpt-3.5-turbo, claude-3-opus, etc.)')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Maximum reflection iterations')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Model temperature')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum tokens per response')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create configuration
    config = {
        'baseline_only': args.baseline_only,
        'dataset': args.dataset,
        'num_tasks': args.tasks,
        'output_dir': args.output_dir,
        'model': args.model,
        'max_iterations': args.max_iterations,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    }

    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()
