"""
Analysis Notebook Script
Interactive analysis of welfare probe experiment results
Can be run as a script or converted to Jupyter notebook
"""

# %% [markdown]
# # Model Welfare Experiments - Analysis Notebook
#
# This notebook provides interactive analysis of the welfare probe experiment results.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries loaded successfully")

# %% [markdown]
# ## 1. Load Results
#
# Load the latest experiment results

# %% Load Data
def load_latest_results(results_dir: str = "results") -> Dict:
    """Load the most recent experiment results"""
    results_path = Path(results_dir)

    # Find latest run directory
    run_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No results found in {results_dir}")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)

    print(f"Loading results from: {latest_run}")

    # Load summary
    summary_file = latest_run / "pipeline_summary.json"
    with open(summary_file) as f:
        data = json.load(f)

    return data

# Load the data
data = load_latest_results()

print(f"Run ID: {data['run_id']}")
print(f"Model: {data['config']['model']}")
print(f"Dataset: {data['config']['dataset']}")
print(f"Tasks: {data['config']['num_tasks']}")

# %% [markdown]
# ## 2. Accuracy Analysis
#
# Compare baseline and modified pipeline accuracy

# %% Accuracy Comparison
baseline_acc = data['evaluation']['accuracy_metrics']['baseline_accuracy']
modified_acc = data['evaluation']['accuracy_metrics']['modified_accuracy']
improvement = data['evaluation']['accuracy_metrics']['improvement']

print(f"Baseline Accuracy:  {baseline_acc:.2%}")
print(f"Modified Accuracy:  {modified_acc:.2%}")
print(f"Improvement:        {improvement:+.2%}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax = axes[0]
bars = ax.bar(['Baseline', 'Modified'],
              [baseline_acc, modified_acc],
              color=['#3498db', '#2ecc71'],
              alpha=0.8,
              edgecolor='black')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Per-task comparison
ax = axes[1]
task_comp = data['evaluation']['accuracy_metrics']['per_task_comparison']
improvements = sum(1 for t in task_comp if t['improvement'])
regressions = sum(1 for t in task_comp if t['regression'])
unchanged = len(task_comp) - improvements - regressions

wedges, texts, autotexts = ax.pie(
    [improvements, regressions, unchanged],
    labels=['Improvements', 'Regressions', 'Unchanged'],
    colors=['#2ecc71', '#e74c3c', '#95a5a6'],
    autopct='%1.1f%%',
    startangle=90
)
ax.set_title('Per-Task Changes', fontsize=14, fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('accuracy_analysis.png', dpi=150)
plt.show()

# %% [markdown]
# ## 3. Welfare Signal Analysis
#
# Analyze welfare signal frequencies and patterns

# %% Welfare Signals
welfare = data['evaluation']['welfare_metrics']

signals = {
    'Overload': welfare['overload_frequency'],
    'Ambiguity': welfare['ambiguity_frequency'],
    'Context Request': welfare['context_request_frequency'],
    'Confidence': welfare['confidence_signal_frequency'],
    'Aversion': welfare['aversion_frequency']
}

# Create dataframe for easier analysis
signals_df = pd.DataFrame.from_dict(signals, orient='index', columns=['Frequency'])
signals_df = signals_df.sort_values('Frequency', ascending=True)

print("Welfare Signal Frequencies:")
print(signals_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Horizontal bar chart
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(signals_df)))
bars = ax.barh(signals_df.index, signals_df['Frequency'], color=colors, edgecolor='black')
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Welfare Signal Frequencies', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.2%}',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Signal correlation heatmap (if detailed results available)
ax = axes[1]
if 'detailed_results' in data['modified']:
    # Extract signal data per task
    signal_matrix = []
    for task_result in data['modified']['detailed_results']:
        welfare_analysis = task_result.get('welfare_analysis', {})
        freqs = welfare_analysis.get('signal_frequencies', {})
        signal_matrix.append([
            freqs.get('overload', 0),
            freqs.get('ambiguity', 0),
            freqs.get('context', 0),
            freqs.get('confidence', 0),
            freqs.get('aversion', 0)
        ])

    signal_matrix = np.array(signal_matrix)

    if len(signal_matrix) > 1:
        # Compute correlation
        corr = np.corrcoef(signal_matrix.T)

        # Plot heatmap
        sns.heatmap(corr,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   xticklabels=['Overload', 'Ambiguity', 'Context', 'Confidence', 'Aversion'],
                   yticklabels=['Overload', 'Ambiguity', 'Context', 'Confidence', 'Aversion'],
                   ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Signal Correlations', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for correlation',
               ha='center', va='center', transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, 'No detailed results available',
           ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('welfare_signals_analysis.png', dpi=150)
plt.show()

# %% [markdown]
# ## 4. Hallucination Analysis
#
# Analyze hallucination detection scores

# %% Hallucination Metrics
hall = data['evaluation']['hallucination_metrics']

print(f"Mean Hallucination Score: {hall['mean_score']:.3f}")
print(f"Std Hallucination Score:  {hall['std_score']:.3f}")
print(f"High Hallucination Count: {hall['high_hallucination_count']}")
print(f"Severe Hallucinations:    {hall['severe_hallucination_count']}")

# Distribution plot
if 'distribution' in hall and hall['distribution']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(hall['distribution'], bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
    ax.axvline(hall['mean_score'], color='darkred', linestyle='--',
              linewidth=2, label=f"Mean: {hall['mean_score']:.2f}")
    ax.axvline(0.5, color='red', linestyle=':',
              linewidth=2, label='High Threshold')
    ax.set_xlabel('Hallucination Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Hallucination Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    box = ax.boxplot([hall['distribution']],
                     labels=['Hallucination Scores'],
                     patch_artist=True,
                     boxprops=dict(facecolor='#f39c12', alpha=0.7),
                     medianprops=dict(color='darkred', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax.axhline(0.5, color='red', linestyle=':', linewidth=2,
              label='High Threshold', alpha=0.7)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Hallucination Score Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('hallucination_analysis.png', dpi=150)
    plt.show()

# %% [markdown]
# ## 5. Efficiency Analysis
#
# Compare computational efficiency

# %% Efficiency Metrics
efficiency = data['evaluation']['efficiency_metrics']

print(f"Baseline Avg Time: {efficiency['baseline_avg_time']:.2f}s")
print(f"Modified Avg Time: {efficiency['modified_avg_time']:.2f}s")
print(f"Time Overhead:     {efficiency['time_overhead']:+.2f}s ({efficiency['time_overhead_percentage']:+.1f}%)")
print(f"\nBaseline Iterations: {efficiency['iteration_comparison']['baseline']:.1f}")
print(f"Modified Iterations: {efficiency['iteration_comparison']['modified']:.1f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time comparison
ax = axes[0]
bars = ax.bar(['Baseline', 'Modified'],
              [efficiency['baseline_avg_time'], efficiency['modified_avg_time']],
              color=['#3498db', '#e67e22'],
              alpha=0.8,
              edgecolor='black')
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Average Completion Time', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Iteration comparison
ax = axes[1]
bars = ax.bar(['Baseline', 'Modified'],
              [efficiency['iteration_comparison']['baseline'],
               efficiency['iteration_comparison']['modified']],
              color=['#3498db', '#9b59b6'],
              alpha=0.8,
              edgecolor='black')
ax.set_ylabel('Average Iterations', fontsize=12)
ax.set_title('Reflection Iterations', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('efficiency_analysis.png', dpi=150)
plt.show()

# %% [markdown]
# ## 6. Statistical Analysis
#
# Test statistical significance of improvements

# %% Statistical Tests
sig = data['evaluation']['accuracy_metrics'].get('statistical_significance', {})

print("Statistical Significance Test:")
print(f"  Test: {sig.get('test', 'N/A')}")
print(f"  P-value: {sig.get('p_value', 1.0):.4f}")
print(f"  Significant: {sig.get('significant', False)}")
print(f"  Alpha: {sig.get('alpha', 0.05)}")

if sig.get('significant', False):
    print("\n✓ The improvement is statistically significant")
else:
    print("\n✗ The improvement is NOT statistically significant")

# %% [markdown]
# ## 7. Summary and Recommendations
#
# Final summary and deployment recommendations

# %% Summary
comp = data['evaluation']['comparative_analysis']

print("="*60)
print("EXPERIMENT SUMMARY")
print("="*60)

print(f"\nAccuracy Improvement: {improvement:+.2%}")
print(f"Cost-Benefit Ratio: {comp['cost_benefit_ratio']:.4f}")
print(f"Overall Improvement: {'Yes' if comp['overall_improvement'] else 'No'}")

print(f"\n{'='*60}")
print("RECOMMENDATION")
print("="*60)
print(f"\n{comp['recommendation']}")
print(f"\n{'='*60}")

# %% [markdown]
# ## 8. Export Results
#
# Export processed results for further analysis

# %% Export
# Create summary dataframe
summary_data = {
    'Metric': [
        'Baseline Accuracy',
        'Modified Accuracy',
        'Improvement',
        'Mean Hallucination Score',
        'Time Overhead',
        'Avg Iterations (Modified)'
    ],
    'Value': [
        f"{baseline_acc:.2%}",
        f"{modified_acc:.2%}",
        f"{improvement:+.2%}",
        f"{hall['mean_score']:.3f}",
        f"{efficiency['time_overhead']:+.2f}s",
        f"{efficiency['iteration_comparison']['modified']:.1f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('experiment_summary.csv', index=False)

print("Summary exported to: experiment_summary.csv")

print("\n✓ Analysis complete!")
