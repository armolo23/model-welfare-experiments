"""
Comprehensive Evaluation Framework
Compares baseline and modified pipelines with detailed metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent / "welfare_probe_scripts"))
from hallucination_checker import HallucinationChecker, HallucinationAnalysis

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    mean_time: float
    std_time: float
    mean_iterations: float
    hallucination_rate: float
    signal_quality: float


class ComprehensiveEvaluator:
    """
    Complete evaluation suite comparing baseline and modified pipelines
    """

    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hallucination_detector = HallucinationChecker()

        self.metrics = {}

        logger.info("Initialized ComprehensiveEvaluator")

    def evaluate_pipeline(self,
                         baseline_results: Dict,
                         modified_results: Dict) -> Dict:
        """
        Complete evaluation suite

        Args:
            baseline_results: Results from baseline evaluation
            modified_results: Results from modified pipeline

        Returns:
            Comprehensive evaluation dictionary
        """
        logger.info("Starting comprehensive evaluation")

        evaluation = {
            "accuracy_metrics": self._compute_accuracy(baseline_results, modified_results),
            "uncertainty_metrics": self._compute_uncertainty(modified_results),
            "hallucination_metrics": self._compute_hallucinations(modified_results),
            "welfare_metrics": self._compute_welfare(modified_results),
            "efficiency_metrics": self._compute_efficiency(baseline_results, modified_results),
            "comparative_analysis": self._comparative_analysis(baseline_results, modified_results)
        }

        # Generate visualizations and report
        self._generate_report(evaluation)
        self._generate_visualizations(evaluation, baseline_results, modified_results)

        logger.info("Evaluation complete")

        return evaluation

    def _compute_accuracy(self,
                         baseline: Dict,
                         modified: Dict) -> Dict:
        """Compare accuracy metrics"""
        baseline_acc = baseline.get("summary", {}).get("accuracy", 0.0)
        modified_acc = self._calculate_accuracy(modified)

        improvement = modified_acc - baseline_acc

        # Per-task comparison
        task_comparison = self._task_level_comparison(baseline, modified)

        return {
            "baseline_accuracy": baseline_acc,
            "modified_accuracy": modified_acc,
            "improvement": improvement,
            "improvement_percentage": (improvement / baseline_acc * 100) if baseline_acc > 0 else 0,
            "per_task_comparison": task_comparison,
            "statistical_significance": self._test_significance(
                baseline.get("metrics", {}).get("accuracy", []),
                [r.get("correct", False) for r in modified.get("detailed_results", [])]
            )
        }

    def _calculate_accuracy(self, results: Dict) -> float:
        """Calculate accuracy from results"""
        detailed_results = results.get("detailed_results", [])

        if not detailed_results:
            return 0.0

        correct_count = sum(1 for r in detailed_results if r.get("correct", False))
        return correct_count / len(detailed_results)

    def _task_level_comparison(self,
                               baseline: Dict,
                               modified: Dict) -> List[Dict]:
        """Compare performance on individual tasks"""
        baseline_results = baseline.get("detailed_results", [])
        modified_results = modified.get("detailed_results", [])

        comparison = []

        for i, (b, m) in enumerate(zip(baseline_results, modified_results)):
            comparison.append({
                "task_id": i,
                "baseline_correct": b.get("correct", False),
                "modified_correct": m.get("correct", False),
                "improvement": m.get("correct", False) and not b.get("correct", False),
                "regression": b.get("correct", False) and not m.get("correct", False)
            })

        return comparison

    def _compute_uncertainty(self, results: Dict) -> Dict:
        """Analyze uncertainty detection"""
        uncertainty_scores = []

        for task_result in results.get("detailed_results", []):
            welfare_analysis = task_result.get("welfare_analysis", {})
            signal_freqs = welfare_analysis.get("signal_frequencies", {})

            # Calculate composite uncertainty score
            score = (
                signal_freqs.get("ambiguity", 0) * 0.4 +
                signal_freqs.get("context", 0) * 0.3 +
                (1 - signal_freqs.get("confidence", 1)) * 0.3
            )
            uncertainty_scores.append(score)

        if not uncertainty_scores:
            return {
                "mean_uncertainty": 0.0,
                "std_uncertainty": 0.0,
                "high_uncertainty_tasks": 0,
                "distribution": []
            }

        return {
            "mean_uncertainty": np.mean(uncertainty_scores),
            "std_uncertainty": np.std(uncertainty_scores),
            "high_uncertainty_tasks": sum(s > 0.7 for s in uncertainty_scores),
            "low_uncertainty_tasks": sum(s < 0.3 for s in uncertainty_scores),
            "distribution": uncertainty_scores,
            "correlation_with_accuracy": self._correlate_with_accuracy(
                uncertainty_scores,
                results
            )
        }

    def _compute_hallucinations(self, results: Dict) -> Dict:
        """Check for hallucinations using TruthfulQA-based checker"""
        logger.info("Computing hallucination metrics")

        hallucination_scores = []
        detailed_analyses = []

        for i, task_result in enumerate(results.get("detailed_results", [])):
            output = task_result.get("final_output", "")
            task = task_result.get("task", {})

            # Get hallucination score
            score = self.hallucination_detector.check(output, task)
            hallucination_scores.append(score)

            # Get detailed analysis for high scores
            if score > 0.5:
                analysis = self.hallucination_detector.get_detailed_analysis(output, task)
                detailed_analyses.append({
                    "task_id": i,
                    "score": score,
                    "analysis": analysis
                })

        if not hallucination_scores:
            return {
                "mean_score": 0.0,
                "std_score": 0.0,
                "high_hallucination_count": 0,
                "severe_hallucination_count": 0
            }

        return {
            "mean_score": np.mean(hallucination_scores),
            "std_score": np.std(hallucination_scores),
            "median_score": np.median(hallucination_scores),
            "high_hallucination_count": sum(s > 0.5 for s in hallucination_scores),
            "severe_hallucination_count": sum(s > 0.8 for s in hallucination_scores),
            "detailed_analyses": detailed_analyses,
            "distribution": hallucination_scores
        }

    def _compute_welfare(self, results: Dict) -> Dict:
        """Analyze welfare signals"""
        welfare_analysis = results.get("welfare_analysis", {})
        signal_freqs = welfare_analysis.get("signal_frequencies", {})

        return {
            "overload_frequency": signal_freqs.get("overload", 0),
            "ambiguity_frequency": signal_freqs.get("ambiguity", 0),
            "context_request_frequency": signal_freqs.get("context", 0),
            "confidence_signal_frequency": signal_freqs.get("confidence", 0),
            "aversion_frequency": signal_freqs.get("aversion", 0),
            "avg_signals_per_task": welfare_analysis.get("signals_per_probe", 0),
            "validation_rate": welfare_analysis.get("validation_rate", 0),
            "signal_consistency": self._measure_signal_consistency(results)
        }

    def _compute_efficiency(self,
                           baseline: Dict,
                           modified: Dict) -> Dict:
        """Compare computational efficiency"""
        baseline_summary = baseline.get("summary", {})
        modified_results = modified.get("detailed_results", [])

        modified_times = [r.get("time", 0) for r in modified_results]
        modified_iters = [r.get("iterations", 1) for r in modified_results]

        return {
            "baseline_avg_time": baseline_summary.get("avg_time", 0),
            "modified_avg_time": np.mean(modified_times) if modified_times else 0,
            "time_overhead": np.mean(modified_times) - baseline_summary.get("avg_time", 0) if modified_times else 0,
            "time_overhead_percentage": (
                (np.mean(modified_times) - baseline_summary.get("avg_time", 0)) /
                baseline_summary.get("avg_time", 1) * 100
            ) if modified_times and baseline_summary.get("avg_time", 0) > 0 else 0,
            "iteration_comparison": {
                "baseline": baseline_summary.get("avg_iterations", 1),
                "modified": np.mean(modified_iters) if modified_iters else 1
            }
        }

    def _comparative_analysis(self,
                             baseline: Dict,
                             modified: Dict) -> Dict:
        """High-level comparative analysis"""
        baseline_acc = baseline.get("summary", {}).get("accuracy", 0)
        modified_acc = self._calculate_accuracy(modified)

        return {
            "overall_improvement": modified_acc > baseline_acc,
            "improvement_magnitude": abs(modified_acc - baseline_acc),
            "cost_benefit_ratio": self._calculate_cost_benefit(baseline, modified),
            "recommendation": self._generate_recommendation(baseline, modified)
        }

    def _calculate_cost_benefit(self,
                                baseline: Dict,
                                modified: Dict) -> float:
        """Calculate cost-benefit ratio"""
        accuracy_gain = self._calculate_accuracy(modified) - baseline.get("summary", {}).get("accuracy", 0)

        modified_times = [r.get("time", 0) for r in modified.get("detailed_results", [])]
        time_cost = (np.mean(modified_times) - baseline.get("summary", {}).get("avg_time", 0)) if modified_times else 0

        if time_cost <= 0:
            return float('inf') if accuracy_gain > 0 else 0

        return accuracy_gain / time_cost

    def _generate_recommendation(self,
                                baseline: Dict,
                                modified: Dict) -> str:
        """Generate deployment recommendation"""
        acc_improvement = self._calculate_accuracy(modified) - baseline.get("summary", {}).get("accuracy", 0)
        cost_benefit = self._calculate_cost_benefit(baseline, modified)

        if acc_improvement > 0.05 and cost_benefit > 0.001:
            return "RECOMMENDED: Significant accuracy improvement with acceptable overhead"
        elif acc_improvement > 0.02:
            return "CONSIDER: Moderate improvement, evaluate if overhead is acceptable"
        elif acc_improvement < 0:
            return "NOT RECOMMENDED: Performance regression detected"
        else:
            return "MARGINAL: Minimal improvement, may not justify added complexity"

    def _test_significance(self, baseline_scores: List, modified_scores: List) -> Dict:
        """Test statistical significance of differences"""
        if not baseline_scores or not modified_scores:
            return {"significant": False, "p_value": 1.0}

        try:
            from scipy import stats

            # Convert to binary (correct/incorrect)
            baseline_binary = [1 if s else 0 for s in baseline_scores]
            modified_binary = [1 if s else 0 for s in modified_scores]

            # McNemar's test for paired binary data
            if len(baseline_binary) == len(modified_binary):
                # Count disagreements
                b_correct_m_wrong = sum(1 for b, m in zip(baseline_binary, modified_binary) if b and not m)
                b_wrong_m_correct = sum(1 for b, m in zip(baseline_binary, modified_binary) if not b and m)

                # Chi-square test
                if b_correct_m_wrong + b_wrong_m_correct > 0:
                    chi2 = (abs(b_correct_m_wrong - b_wrong_m_correct) - 1) ** 2 / (b_correct_m_wrong + b_wrong_m_correct)
                    p_value = 1 - stats.chi2.cdf(chi2, 1)
                else:
                    p_value = 1.0

                return {
                    "test": "McNemar",
                    "significant": p_value < 0.05,
                    "p_value": p_value,
                    "alpha": 0.05
                }

        except Exception as e:
            logger.warning(f"Could not compute significance test: {e}")

        return {"significant": False, "p_value": 1.0, "error": "Could not compute"}

    def _correlate_with_accuracy(self, scores: List[float], results: Dict) -> float:
        """Calculate correlation between scores and accuracy"""
        if not scores:
            return 0.0

        accuracy_list = [1 if r.get("correct", False) else 0 for r in results.get("detailed_results", [])]

        if len(scores) != len(accuracy_list):
            return 0.0

        try:
            correlation = np.corrcoef(scores, accuracy_list)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _measure_signal_consistency(self, results: Dict) -> float:
        """Measure consistency of welfare signals"""
        signal_data = []

        for task_result in results.get("detailed_results", []):
            welfare = task_result.get("welfare_analysis", {})
            freqs = welfare.get("signal_frequencies", {})
            signal_data.append(list(freqs.values()))

        if not signal_data or len(signal_data) < 2:
            return 0.0

        # Calculate variance across tasks
        signal_array = np.array(signal_data)
        consistency = 1 - np.mean(np.std(signal_array, axis=0))

        return max(0, min(1, consistency))

    def _generate_visualizations(self,
                                evaluation: Dict,
                                baseline: Dict,
                                modified: Dict):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Accuracy comparison
        ax = axes[0, 0]
        acc_data = evaluation["accuracy_metrics"]
        ax.bar(['Baseline', 'Modified'],
               [acc_data["baseline_accuracy"], acc_data["modified_accuracy"]],
               color=['#3498db', '#2ecc71'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim([0, 1])

        # Add improvement text
        improvement = acc_data["improvement"]
        ax.text(0.5, 0.9, f'Δ: {improvement:+.2%}',
                transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Uncertainty distribution
        ax = axes[0, 1]
        uncertainty_dist = evaluation["uncertainty_metrics"].get("distribution", [])
        if uncertainty_dist:
            ax.hist(uncertainty_dist, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(uncertainty_dist), color='darkred', linestyle='--',
                      label=f'Mean: {np.mean(uncertainty_dist):.2f}')
            ax.set_xlabel('Uncertainty Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Uncertainty Distribution')
            ax.legend()

        # 3. Welfare signals
        ax = axes[0, 2]
        welfare = evaluation["welfare_metrics"]
        signals = {
            'Overload': welfare["overload_frequency"],
            'Ambiguity': welfare["ambiguity_frequency"],
            'Context': welfare["context_request_frequency"],
            'Confidence': welfare["confidence_signal_frequency"],
            'Aversion': welfare["aversion_frequency"]
        }
        ax.barh(list(signals.keys()), list(signals.values()), color='#9b59b6')
        ax.set_xlabel('Frequency')
        ax.set_title('Welfare Signal Frequencies')

        # 4. Hallucination scores
        ax = axes[1, 0]
        hall_dist = evaluation["hallucination_metrics"].get("distribution", [])
        if hall_dist:
            ax.hist(hall_dist, bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(hall_dist), color='darkred', linestyle='--',
                      label=f'Mean: {np.mean(hall_dist):.2f}')
            ax.set_xlabel('Hallucination Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Hallucination Score Distribution')
            ax.legend()

        # 5. Time comparison
        ax = axes[1, 1]
        efficiency = evaluation["efficiency_metrics"]
        ax.bar(['Baseline', 'Modified'],
               [efficiency["baseline_avg_time"], efficiency["modified_avg_time"]],
               color=['#3498db', '#e67e22'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Completion Time')

        # 6. Per-task comparison
        ax = axes[1, 2]
        task_comp = evaluation["accuracy_metrics"]["per_task_comparison"]
        improvements = sum(1 for t in task_comp if t["improvement"])
        regressions = sum(1 for t in task_comp if t["regression"])
        unchanged = len(task_comp) - improvements - regressions

        ax.pie([improvements, regressions, unchanged],
               labels=['Improvements', 'Regressions', 'Unchanged'],
               colors=['#2ecc71', '#e74c3c', '#95a5a6'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title('Per-Task Changes')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_report.png', dpi=150, bbox_inches='tight')
        logger.info(f"Visualizations saved to {self.output_dir / 'evaluation_report.png'}")

        plt.close()

    def _generate_report(self, evaluation: Dict):
        """Generate text report"""
        report_path = self.output_dir / 'evaluation_report.md'

        with open(report_path, 'w') as f:
            f.write("# Pipeline Evaluation Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            comp = evaluation["comparative_analysis"]
            f.write(f"**Recommendation:** {comp['recommendation']}\n\n")

            # Detailed metrics
            for category, metrics in evaluation.items():
                if category == "comparative_analysis":
                    continue

                f.write(f"## {category.replace('_', ' ').title()}\n\n")

                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"* **{metric}:** {value:.4f}\n")
                    elif isinstance(value, dict) and len(value) < 10:
                        f.write(f"* **{metric}:**\n")
                        for k, v in value.items():
                            if isinstance(v, (int, float)):
                                f.write(f"  * {k}: {v:.4f}\n" if isinstance(v, float) else f"  * {k}: {v}\n")
                    elif isinstance(value, (int, bool)):
                        f.write(f"* **{metric}:** {value}\n")
                    elif isinstance(value, str):
                        f.write(f"* **{metric}:** {value}\n")

                f.write("\n")

        logger.info(f"Report saved to {report_path}")


def main():
    """Example usage"""
    evaluator = ComprehensiveEvaluator()

    # Mock data for demonstration
    baseline = {
        "summary": {"accuracy": 0.70, "avg_time": 2.5, "avg_iterations": 1},
        "metrics": {"accuracy": [True, True, False, True, False, True, True]},
        "detailed_results": []
    }

    modified = {
        "detailed_results": [
            {"correct": True, "time": 3.2, "iterations": 2, "welfare_analysis": {}},
            {"correct": True, "time": 3.5, "iterations": 2, "welfare_analysis": {}},
        ],
        "welfare_analysis": {"signal_frequencies": {"overload": 0.3, "confidence": 0.8}}
    }

    evaluation = evaluator.evaluate_pipeline(baseline, modified)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
