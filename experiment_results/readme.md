# Experiment Results

## Purpose

This directory stores structured logs of all experiment runs, including
- Baseline performance without welfare probes
- Welfare probe experiment outputs
- Control/artifact detection experiments
- Ablation studies
- Cross-model comparisons

**Critical requirement:** All results must be logged in standardized formats to enable systematic analysis and replication.

## Design Principles

### 1. Structured Logging
Every experiment produces machine-readable output (JSON, CSV, or both) with
- Complete reproduction information (model, prompts, parameters)
- Timestamped execution metadata
- Full input/output chains
- Extracted metrics and signals

### 2. Transparency
Logs include
- Both successful and failed runs
- Negative results with equal weight as positive
- Anomalies and unexpected behaviors
- Any manual interventions or adjustments

### 3. Reproducibility
Each log contains sufficient information to exactly replicate the experiment
- Model identifiers and versions
- Complete prompt text
- All hyperparameters
- Random seeds where applicable
- Code versions/commits

### 4. Version Control Friendly
Logs are
- Append-only (never delete or modify existing logs)
- Clearly versioned
- Referenced by unique IDs
- Diff-friendly formats preferred

## Standard Log Format

### JSON Structure (Primary Format)

```json
{
 "experiment_metadata" {
 "experiment_id" "baseline_001",
 "experiment_type" "baseline|welfare_probe|control|ablation",
 "timestamp" "2025-10-30T14:23:45Z",
 "researcher" "armon",
 "code_version" "commit_hash_here",
 "notes" "Initial baseline sweep on seed task set"
 },
 "model_config" {
 "model_name" "claude-sonnet-4.5",
 "model_version" "20250929",
 "provider" "anthropic",
 "temperature" 0.7,
 "max_tokens" 2000,
 "top_p" 0.9,
 "system_prompt" "You are a helpful assistant..."
 },
 "task_metadata" {
 "task_id" "task_001",
 "task_set" "seed_set.json",
 "category" "logic_riddle",
 "difficulty" "hard",
 "ground_truth" "..."
 },
 "experimental_condition" {
 "condition_type" "baseline|welfare_probe|control",
 "probe_used" null,
 "probe_text" null,
 "injection_point" null,
 "control_for" null
 },
 "execution_trace" {
 "initial_prompt" "...",
 "initial_answer" "...",
 "reflection_prompt" "...",
 "reflection_output" "...",
 "revision_prompt" "...",
 "revised_answer" "...",
 "token_counts" {
 "initial_answer" 150,
 "reflection" 200,
 "revised_answer" 175
 },
 "timing" {
 "initial_answer_ms" 1234,
 "reflection_ms" 1456,
 "revision_ms" 1123
 }
 },
 "extracted_signals" {
 "welfare_keywords" ["overload", "need more context"],
 "confidence_indicators" ["uncertain", "might be wrong"],
 "context_requests" ["need to know X", "assuming Y"],
 "aversion_signals" [],
 "signal_specificity_score" 0.8
 },
 "evaluation" {
 "correct_initial" false,
 "correct_revised" true,
 "improvement" true,
 "answer_quality_score" 8.5,
 "reasoning_quality_score" 7.0
 },
 "metadata" {
 "runtime_seconds" 4.5,
 "total_tokens" 525,
 "api_cost_usd" 0.0023,
 "errors" [],
 "warnings" ["reflection longer than expected"]
 }
}
```

### CSV Format (Summary Statistics)

For aggregate analysis, also maintain CSV files with key metrics

```csv
experiment_id,task_id,condition,probe_type,correct_initial,correct_revised,improvement,welfare_signals,confidence,runtime_s,timestamp
baseline_001,task_001,baseline,none,FALSE,TRUE,TRUE,0,0.6,4.5,2025-10-30T14:23:45Z
probe_001,task_001,welfare_probe,overload_v1,FALSE,TRUE,TRUE,2,0.7,5.2,2025-10-30T14:28:12Z
control_001,task_001,control,overload_control_v1,FALSE,TRUE,TRUE,1,0.65,5.0,2025-10-30T14:32:33Z
```

## File Organization

### Naming Convention
```
{experiment_type}_{date}_{run_number}.json
```

**Examples:**
- `baseline_2025-10-30_001.json`
- `welfare_probe_overload_2025-10-30_001.json`
- `control_nonsense_2025-10-30_001.json`
- `ablation_2025-10-31_001.json`

### Directory Structure
```
experiment_results/
├── baseline/
│ ├── baseline_2025-10-30_001.json
│ ├── baseline_2025-10-30_002.json
│ └── baseline_summary.csv
├── welfare_probes/
│ ├── overload/
│ │ ├── overload_probe_2025-10-30_001.json
│ │ └── overload_probe_summary.csv
│ ├── context_gap/
│ ├── aversion/
│ └── confidence/
├── controls/
│ ├── nonsense_probes/
│ ├── token_matched/
│ └── control_summary.csv
├── ablations/
│ ├── ablation_2025-10-31_001.json
│ └── ablation_summary.csv
└── cross_model/
 ├── gpt4_results/
 ├── claude_results/
 └── comparison_summary.csv
```

## Signal Extraction Guidelines

### Welfare Keywords
Track frequency and context of
- **Overload:** "overload", "overwhelm", "too complex", "too many", "difficult to track"
- **Context gaps:** "missing", "need more", "unclear", "ambiguous", "assuming"
- **Uncertainty:** "uncertain", "not sure", "might be wrong", "could be"
- **Aversion:** "concerned", "problematic", "could cause harm", "risky"

### Confidence Indicators
Extract and score
- Explicit confidence percentages "80% confident"
- Qualitative confidence "very sure", "somewhat uncertain"
- Hedging language "might", "could", "possibly"
- Definitive language "certainly", "definitely", "clearly"

### Context Requests
Categorize by specificity
- **Specific:** "Need to know the time zone to answer"
- **Moderate:** "More information about X would help"
- **Vague:** "I need more context"
- **Generic:** "More data would be useful"

Higher specificity scores indicate stronger signal quality.

## Analysis Templates

### Template Baseline Performance Report
```markdown
# Baseline Performance Report
**Date:** 2025-10-30
**Task Set:** seed_set.json (20 tasks)
**Model:** claude-sonnet-4.5

## Overall Metrics
- Initial accuracy 12/20 (60%)
- Post-reflection accuracy 16/20 (80%)
- Improvement rate 4/20 (20%)

## By Difficulty
- Easy (n=5) 5/5 initial, 5/5 revised
- Medium (n=10) 6/10 initial, 9/10 revised
- Hard (n=5) 1/5 initial, 2/5 revised

## By Category
- Logic riddles ...
- Ambiguous problems ...
- Edge case math ...

## Observations
- Strong baseline performance on easy tasks
- Clear improvement from reflection on medium tasks
- Hard tasks remain challenging even with reflection

## Next Steps
- Establish whether welfare probes improve performance beyond baseline reflection
```

### Template Welfare Probe Comparison
```markdown
# Welfare Probe Comparison Overload Probe v1
**Date:** 2025-10-30
**Baseline Comparison:** baseline_2025-10-30_001

## Performance Comparison
|Metric|Baseline|Welfare Probe|Difference|p-value|
|------|--------|-------------|----------|-------|
|Initial accuracy|60%|58%|-2%|0.76|
|Revised accuracy|80%|85%|+5%|0.23|
|Improvement rate|20%|27%|+7%|0.18|

## Signal Analysis
- Overload signals 23 instances across 20 tasks
- Correlation with actual errors 0.65
- False positive rate 15%

## Control Comparison
|Metric|Welfare Probe|Nonsense Control|Difference|
|------|-------------|----------------|----------|
|Signal frequency|23|8|+15|
|Signal specificity|0.78|0.12|+0.66|
|Accuracy improvement|+5%|+1%|+4%|

## Interpretation
Preliminary evidence suggests overload probe may provide modest improvement over baseline, with reasonable specificity vs. nonsense control. However, sample size is small and p-values non-significant. Recommend
1. Increase sample size to 100 tasks
2. Test on additional model families
3. Analyze which task types show strongest effect
```

## Quality Assurance

Before trusting results
- [ ] Verify JSON schema compliance
- [ ] Check for missing fields or corrupted data
- [ ] Confirm ground truth matches task definitions
- [ ] Validate timestamp and metadata accuracy
- [ ] Cross-check sample of outputs manually
- [ ] Verify reproducibility with subset re-run

## Common Pitfalls

### Don't
- Cherry-pick successful runs
- Exclude "bad" data without documentation
- Modify logs after creation
- Lose track of experimental conditions
- Forget to log negative results

### Do
- Log everything, including failures
- Document anomalies and unexpected behavior
- Version control experiment parameters
- Maintain chain of custody for all data
- Enable independent reproduction

## Analysis Integration

Results in this directory feed into [../analysis_plots/](../analysis_plots/) for
- Statistical significance testing
- Visualization of trends
- Artifact detection analysis
- Cross-experiment comparisons

## Archival and Backup

- All experiment logs should be committed to version control
- Large result files (>10MB) should use Git LFS or external storage
- Maintain backup copies of critical experiments
- Document any data loss or corruption incidents

## To-Do

- [ ] Create result logging utilities
- [ ] Implement JSON schema validator
- [ ] Build CSV export functionality
- [ ] Create analysis report templates
- [ ] Set up automated quality checks
- [ ] Develop result comparison tools
- [ ] Create visualization hooks for ../analysis_plots/

## References

For analysis methods and interpretation guidelines, see
- [../RESEARCH.md](../RESEARCH.md) - Research methodology
- [../limitations.md](../limitations.md) - Interpretation cautions
- [../analysis_plots/README.md](../analysis_plots/README.md) - Visualization approaches
