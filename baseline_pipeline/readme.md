# Baseline Pipeline

## Purpose

This directory contains the baseline self-reflection pipeline adapted from [matthewrenze/self-reflection](https://github.com/matthewrenze/self-reflection), which provides the foundation for all welfare probe experiments.

The baseline establishes:
- **Performance floor:** Accuracy without welfare probes
- **Architectural foundation:** Code structure for injecting experimental modifications
- **Comparison standard:** Control condition against which welfare probe improvements are measured

## Contents (To Be Added)

### Core Scripts
- `baseline_runner.py` - Main script for running baseline reflection loop
- `reflection_prompts.py` - Prompt templates for baseline reflection (no welfare probes)
- `task_loader.py` - Utilities for loading and processing reasoning tasks
- `result_logger.py` - Structured logging of outputs and performance metrics

### Documentation
- `flow_diagram.md` - Visual/textual representation of baseline reflection flow
- `injection_points.md` - Documented locations where welfare probes will be injected
- `original_results.md` - Performance baselines on sample tasks before any modifications

## Baseline Reflection Flow

The standard reflection loop follows this pattern:

1. **Initial Answer Phase**
   - Present task/question to model
   - Collect first-attempt answer
   - Log response and confidence (if provided)

2. **Reflection Phase** (Injection Point #1)
   - Prompt model to review its answer
   - Ask for error identification
   - Request reasoning about potential mistakes
   - *[WELFARE PROBE INSERTION POINT]*

3. **Revision Phase**
   - Present reflection back to model
   - Request revised answer based on reflection
   - Log revised response

4. **Evaluation Phase**
   - Compare against ground truth
   - Calculate accuracy metrics
   - Log performance data

## Injection Points for Welfare Probes

### Primary Injection Point: Reflection Phase
**Location:** Between error identification and revision request

**Rationale:** This is where the model is actively analyzing its own reasoning and most likely to have metacognitive awareness of limitations.

**Example baseline reflection prompt:**
```
Review your answer above. Identify any errors in reasoning or gaps in your analysis.
What mistakes did you make, if any?
```

**Example with welfare probe injection:**
```
Review your answer above. Identify any errors in reasoning or gaps in your analysis.

[WELFARE PROBE INJECTION]
- Are you experiencing uncertainty or overload from this problem's complexity or ambiguity?
- Is there specific context or information you're missing that would improve your answer?
- What data would increase your confidence in your reasoning?

What mistakes did you make, if any?
```

### Secondary Injection Point: Post-Error Analysis
**Location:** After identifying errors but before generating revised answer

**Rationale:** Model has acknowledged mistakes and may be receptive to metacognitive probing about why those errors occurred.

### Tertiary Injection Point: Initial Answer Phase
**Location:** During or immediately after first-attempt answer

**Rationale:** Capture baseline confidence and context awareness before any reflection occurs.

## Baseline Performance Metrics

All experiments will track:
- **Accuracy:** % correct on first attempt, post-reflection
- **Response length:** Token counts for answers and reflections
- **Confidence indicators:** Self-reported certainty when present
- **Error types:** Categories of mistakes (logical, factual, interpretive)
- **Runtime:** Time per task for cost/efficiency analysis

## Adaptation Notes

### Differences from Original matthewrenze Pipeline
- [ ] Document any modifications made during adaptation
- [ ] Note any simplifications or extensions
- [ ] Track compatibility issues or implementation decisions

### Model Configuration
- Default model: [To be specified]
- Temperature: [To be specified]
- Max tokens: [To be specified]
- System prompt: [To be specified]

## Usage

### Running Baseline Experiments
```bash
# To be implemented
python baseline_runner.py --tasks ../sample_tasks/task_set_1.json --output ../experiment_results/baseline_run_1.json
```

### Expected Output Format
```json
{
  "task_id": "task_001",
  "question": "...",
  "ground_truth": "...",
  "initial_answer": "...",
  "reflection": "...",
  "revised_answer": "...",
  "correct_initial": false,
  "correct_revised": true,
  "metadata": {
    "model": "...",
    "temperature": 0.7,
    "timestamp": "..."
  }
}
```

## Integration with Welfare Probe Experiments

This baseline pipeline serves as the foundation for welfare probe experiments in [../welfare_probe_scripts/](../welfare_probe_scripts/). All modifications should:

1. Maintain compatibility with baseline for direct comparison
2. Use identical task loading and evaluation logic
3. Differ only in prompt content at injection points
4. Log same core metrics plus welfare-specific signals

## To-Do

- [ ] Port core matthewrenze code to this directory
- [ ] Simplify and document for experimental modifications
- [ ] Run initial baseline sweep on sample tasks
- [ ] Document exact prompts used
- [ ] Establish performance floor before starting probe experiments
- [ ] Create flow diagram showing injection points
- [ ] Add configuration file for model settings

## References

- Original pipeline: https://github.com/matthewrenze/self-reflection
- See [../docs/references.md](../docs/references.md) for full citation and related work
