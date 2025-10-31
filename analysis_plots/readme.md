# Analysis and Visualization

## Purpose

This directory contains scripts, notebooks, and generated visualizations for analyzing welfare probe experiment results. The focus is on
1. **Detecting artifacts** and distinguishing signal from noise
2. **Statistical rigor** in measuring performance differences
3. **Transparency** in showing both positive and negative results
4. **Interpretability** through clear visualizations

All analysis should be reproducible, well-documented, and resistant to confirmation bias.

## Analysis Principles

### 1. Pre-Registration
Define metrics and interpretation criteria before running experiments
- What constitutes success/failure?
- What effect sizes are meaningful?
- How will we correct for multiple comparisons?

### 2. Artifact Detection First
Every positive signal must be tested against
- Nonsense control conditions
- Token-matched baselines
- Position/order effects
- Cross-model consistency

### 3. Statistical Honesty
- Report confidence intervals, not just point estimates
- Correct for multiple testing
- Don't p-hack or cherry-pick analyses
- Report negative results with same rigor as positive

### 4. Visualization Clarity
Plots should
- Show raw data points, not just summaries
- Include error bars and uncertainty
- Highlight negative controls
- Be interpretable without prior knowledge

## Core Analysis Types

### 1. Performance Comparison Analysis

**Objective:** Measure whether welfare probes improve accuracy over baseline

**Metrics:**
- Initial accuracy (before reflection)
- Post-reflection accuracy
- Improvement rate (% tasks that got better)
- Performance by task difficulty and category

**Statistical tests:**
- Paired t-test or Wilcoxon signed-rank (baseline vs. probe on same tasks)
- Bootstrap confidence intervals
- Effect size (Cohen's d)
- Multiple testing correction (Bonferroni or Benjamini-Hochberg)

**Visualizations:**
- Accuracy comparison bar charts with error bars
- Task-by-task improvement scatter plots
- Difficulty stratified comparisons
- Box plots showing distribution of scores

### 2. Artifact Detection Analysis

**Objective:** Distinguish genuine welfare signals from prompt artifacts

**Comparisons:**
- Welfare probe vs. nonsense probe (e.g., "overload" vs. "blue overload")
- Welfare probe vs. token-matched non-welfare prompt
- Specific requests vs. generic requests

**Metrics:**
- Signal frequency (how often keywords appear)
- Signal specificity score (relevant vs. generic)
- False positive rate (signals without errors)
- False negative rate (errors without signals)

**Visualizations:**
- Side-by-side frequency comparison (real vs. nonsense probes)
- Scatter plot signal frequency vs. task difficulty
- Confusion matrix signals vs. actual errors
- Specificity score distributions

### 3. Consistency Analysis

**Objective:** Measure whether welfare signals are reliable across runs and models

**Tests:**
- Same task, multiple runs do signals repeat?
- Same task, different models do signals generalize?
- Rephrased probes are signals sensitive to wording?

**Metrics:**
- Inter-run correlation
- Cross-model agreement rate
- Prompt sensitivity index

**Visualizations:**
- Consistency heatmaps (tasks × runs)
- Cross-model comparison matrices
- Prompt variation sensitivity plots

### 4. Signal Content Analysis

**Objective:** Analyze what models actually say in welfare probe responses

**Methods:**
- Keyword frequency analysis
- Context request specificity scoring
- Semantic clustering of similar responses
- Manual categorization of signal types

**Metrics:**
- Top welfare keywords by frequency
- Specificity vs. vagueness ratio
- Context request relevance scores
- Signal diversity index

**Visualizations:**
- Word clouds (used cautiously)
- Keyword frequency bar charts
- Specificity distribution histograms
- Response category breakdowns

### 5. Correlation Analysis

**Objective:** Test relationships between welfare signals and performance

**Questions:**
- Do tasks with "overload" signals have lower accuracy?
- Do specific context requests correlate with improvement?
- Do confidence ratings predict actual correctness?

**Statistical tests:**
- Pearson or Spearman correlation
- Logistic regression (predicting correctness from signals)
- ROC curves for predictive power

**Visualizations:**
- Scatter plots with regression lines
- ROC curves for signal predictive value
- Correlation matrices
- Stratified analysis by task type

### 6. Ablation Analysis

**Objective:** Identify which probe components drive effects

**Design:**
- Baseline (no probe)
- Individual probe questions
- Combinations of questions
- Full probe set

**Metrics:**
- Marginal contribution of each component
- Interaction effects
- Diminishing returns curve

**Visualizations:**
- Component contribution bar charts
- Interaction effect plots
- Cumulative improvement curves

## Visualization Catalog

### Standard Plots

#### 1. Accuracy Comparison Plot
```python
# Bar chart baseline vs. welfare probe vs. control
# Error bars 95% CI from bootstrap
# Separate bars for initial and post-reflection accuracy
# Annotations p-values, effect sizes
```

#### 2. Improvement Scatter Plot
```python
# X-axis baseline improvement (0 or 1)
# Y-axis welfare probe improvement (0 or 1)
# Points individual tasks
# Diagonal line equal performance
# Quadrants both improve, only baseline, only probe, neither
```

#### 3. Artifact Detection Comparison
```python
# Side-by-side bar charts
# Left welfare probe signal frequency
# Right nonsense control signal frequency
# Color coding signal types
# Statistical test annotation
```

#### 4. Consistency Heatmap
```python
# Rows tasks
# Columns experimental runs
# Cell color signal present/absent or accuracy
# Patterns reveal consistency vs. noise
```

#### 5. Signal Specificity Distribution
```python
# Histogram of specificity scores
# Separate distributions for welfare vs. control probes
# Overlaid to show separation (or lack thereof)
```

#### 6. ROC Curve for Signal Predictivity
```python
# True positive rate vs. false positive rate
# Can welfare signals predict which tasks will have errors?
# AUC score indicates predictive power
```

#### 7. Cross-Model Comparison Matrix
```python
# Models as rows and columns
# Cells agreement rate on welfare signals
# Diagonal self-consistency
# Off-diagonal cross-model generalization
```

#### 8. Keyword Frequency Analysis
```python
# Bar chart top welfare keywords
# Separate bars for welfare probe, control probe, baseline
# Helps identify artifact keywords vs. genuine signals
```

## Jupyter Notebooks

### Primary Analysis Notebooks

**`01_baseline_analysis.ipynb`**
- Load baseline results
- Compute performance metrics
- Establish floor for comparison
- Identify task difficulty patterns

**`02_welfare_probe_comparison.ipynb`**
- Load welfare probe results
- Direct comparison with baseline
- Statistical significance testing
- Effect size calculations

**`03_artifact_detection.ipynb`**
- Compare welfare probes to nonsense controls
- Token-matched baseline comparisons
- Signal specificity scoring
- False positive/negative analysis

**`04_consistency_analysis.ipynb`**
- Cross-run consistency
- Cross-model replication
- Prompt sensitivity testing
- Reliability metrics

**`05_signal_content_analysis.ipynb`**
- Keyword extraction and frequency
- Context request categorization
- Specificity vs. vagueness scoring
- Qualitative pattern identification

**`06_correlation_analysis.ipynb`**
- Signal-performance correlations
- Predictive modeling
- ROC curves and calibration
- Subgroup analyses

**`07_ablation_analysis.ipynb`**
- Component-wise contribution
- Interaction effects
- Minimal effective probe identification

**`08_negative_results.ipynb`**
- Document what didn't work
- Failed probes and dead ends
- Anti-patterns identified
- Lessons learned

### Supporting Notebooks

**`exploratory_analysis.ipynb`**
- Open-ended exploration
- Hypothesis generation
- Sanity checks
- Quick visualizations

**`quality_assurance.ipynb`**
- Data validation
- Missing data checks
- Outlier detection
- Consistency verification

## Python Scripts

### Analysis Utilities

**`metrics.py`**
- Functions for computing accuracy, improvement, etc.
- Statistical test implementations
- Effect size calculations
- Confidence interval estimation

**`visualization.py`**
- Plotting functions for standard visualizations
- Consistent styling and formatting
- Annotation helpers
- Export utilities

**`artifact_tests.py`**
- Automated artifact detection
- Nonsense probe comparisons
- Token-matching controls
- Position effect tests

**`signal_extraction.py`**
- Parse welfare signals from logs
- Keyword frequency counting
- Specificity scoring
- Context request categorization

**`statistical_tests.py`**
- Hypothesis testing functions
- Multiple comparison corrections
- Bootstrap resampling
- Permutation tests

**`report_generation.py`**
- Automated report creation
- Template-based summaries
- Result aggregation
- Export to markdown/HTML

## Analysis Workflow

### Standard Analysis Pipeline

1. **Load and Validate Data**
 ```python
 baseline = load_results("../experiment_results/baseline/")
 probe = load_results("../experiment_results/welfare_probes/overload/")
 control = load_results("../experiment_results/controls/")
 validate_data([baseline, probe, control])
 ```

2. **Compute Core Metrics**
 ```python
 baseline_metrics = compute_metrics(baseline)
 probe_metrics = compute_metrics(probe)
 control_metrics = compute_metrics(control)
 ```

3. **Statistical Comparison**
 ```python
 comparison = compare_conditions(baseline_metrics, probe_metrics)
 artifact_test = compare_conditions(probe_metrics, control_metrics)
 ```

4. **Visualization**
 ```python
 plot_accuracy_comparison(baseline_metrics, probe_metrics, control_metrics)
 plot_artifact_detection(probe_metrics, control_metrics)
 plot_consistency_heatmap(probe)
 ```

5. **Report Generation**
 ```python
 generate_report(comparison, artifact_test, output="reports/overload_probe_analysis.md")
 ```

## Quality Checks

### Before Trusting Results

- [ ] Verify data completeness (no missing experiments)
- [ ] Check for data quality issues (corrupted logs, parsing errors)
- [ ] Confirm matched task sets across conditions
- [ ] Validate ground truth consistency
- [ ] Inspect outliers and anomalies
- [ ] Cross-check sample calculations manually
- [ ] Review statistical assumptions (normality, independence)
- [ ] Verify multiple testing corrections applied

### Red Flags

Watch for signs of unreliable results
- Too-good-to-be-true effect sizes
- Perfect separation between conditions
- Patterns that reverse with small changes
- High sensitivity to analysis choices
- Results that contradict controls
- Inconsistency across model families

## Interpretation Guidelines

### What Constitutes Convincing Evidence?

**Strong evidence for welfare probe effectiveness:**
- Statistically significant accuracy improvement (p < 0.01)
- Meaningful effect size (Cohen's d > 0.5)
- Clear separation from nonsense controls
- Consistent across multiple runs
- Replicates across model families
- Specific, relevant context requests
- Correlation between signals and task characteristics

**Weak or inconclusive evidence:**
- Small effect sizes (Cohen's d < 0.3)
- High variance across runs
- Marginal statistical significance (p ~ 0.05)
- Similar patterns in nonsense controls
- Model-specific artifacts
- Generic, vague signals
- Cherry-picked examples

**Evidence against hypothesis:**
- No performance difference vs. baseline
- Indistinguishable from nonsense controls
- Inconsistent across runs
- Signals don't correlate with errors
- Cross-model divergence
- High false positive rate

### Communicating Results

**Always report:**
- Effect sizes with confidence intervals
- Raw data visualizations, not just summaries
- Negative control comparisons
- Failed experiments and null results
- Limitations and alternative explanations

**Never:**
- Cherry-pick favorable results
- Hide negative findings
- Overinterpret weak effects
- Claim certainty where uncertainty exists
- Ignore contradictory evidence

## Output Organization

```
analysis_plots/
├── figures/
│ ├── accuracy_comparison.png
│ ├── artifact_detection.png
│ ├── consistency_heatmap.png
│ └── ...
├── reports/
│ ├── baseline_report.md
│ ├── overload_probe_report.md
│ ├── negative_results_report.md
│ └── ...
├── notebooks/
│ ├── 01_baseline_analysis.ipynb
│ ├── 02_welfare_probe_comparison.ipynb
│ └── ...
└── scripts/
 ├── metrics.py
 ├── visualization.py
 ├── artifact_tests.py
 └── ...
```

## To-Do

- [ ] Implement core metrics functions
- [ ] Create standard visualization templates
- [ ] Build artifact detection pipeline
- [ ] Develop statistical testing utilities
- [ ] Create analysis notebooks
- [ ] Set up automated report generation
- [ ] Design quality assurance checks
- [ ] Document interpretation guidelines

## References

For statistical methods and best practices
- [../RESEARCH.md](../RESEARCH.md) - Research methodology
- [../limitations.md](../limitations.md) - Interpretation cautions
- Statistical best practices in psychology and ML literature
- Visualization principles (Tufte, Cairo, Wilke)
