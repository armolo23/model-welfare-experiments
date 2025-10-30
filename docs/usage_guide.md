# Usage Guide: Model Welfare Experiments

Complete guide for running welfare probe experiments on language models.

---

## Quick Start

### 1. Environment Setup

**Windows:**
```batch
setup.bat
```

**Unix/Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Configure API Keys

Edit the `.env` file created during setup:

```bash
# .env
OPENAI_API_KEY=your_actual_key_here
ANTHROPIC_API_KEY=your_actual_key_here

MODEL_NAME=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=500
```

### 3. Run Your First Experiment

**Baseline only (quick test):**
```bash
python run_pipeline.py --baseline-only --tasks 5
```

**Full pipeline:**
```bash
python run_pipeline.py --tasks 10
```

---

## Detailed Usage

### Running Experiments

#### Command-Line Options

```bash
python run_pipeline.py [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--baseline-only` | Run baseline evaluation only | False |
| `--dataset` | Dataset to use (hotpotqa, alfworld, custom) | hotpotqa |
| `--tasks` | Number of tasks to evaluate | 20 |
| `--output-dir` | Output directory for results | results |
| `--model` | Model to use | gpt-4 |
| `--max-iterations` | Maximum reflection iterations | 3 |
| `--temperature` | Model temperature | 0.7 |
| `--max-tokens` | Maximum tokens per response | 500 |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

#### Example Commands

**Quick test with 5 tasks:**
```bash
python run_pipeline.py --tasks 5 --model gpt-3.5-turbo
```

**Full evaluation with HotpotQA:**
```bash
python run_pipeline.py --dataset hotpotqa --tasks 20
```

**Custom output directory:**
```bash
python run_pipeline.py --output-dir experiments/run1 --tasks 15
```

**Using Claude:**
```bash
python run_pipeline.py --model claude-3-opus-20240229 --tasks 10
```

**Debugging mode:**
```bash
python run_pipeline.py --tasks 5 --log-level DEBUG
```

---

## Understanding Results

### Output Directory Structure

After running, results are organized as:

```
results/
└── 20250130_143022/          # Run ID (timestamp)
    ├── baseline/
    │   ├── baseline_results.json
    │   └── config.json
    ├── modified/
    │   ├── modified_results.json
    │   └── config.json
    ├── evaluation/
    │   ├── evaluation_results.json
    │   ├── evaluation_report.md
    │   └── evaluation_report.png
    └── pipeline_summary.json
```

### Key Result Files

**`pipeline_summary.json`**
- Complete experiment summary
- Baseline and modified results
- Evaluation metrics
- Configuration

**`evaluation_report.md`**
- Human-readable analysis
- Accuracy comparison
- Welfare signal frequencies
- Hallucination metrics
- Recommendations

**`evaluation_report.png`**
- Visual summary with 6 plots:
  - Accuracy comparison
  - Uncertainty distribution
  - Welfare signal frequencies
  - Hallucination scores
  - Time comparison
  - Per-task changes

---

## Analysis

### Interactive Analysis

Run the analysis notebook:

```bash
# As a Python script
python analysis_plots/analysis_notebook.py

# Or convert to Jupyter notebook
pip install jupytext
jupytext --to notebook analysis_plots/analysis_notebook.py
jupyter notebook analysis_plots/analysis_notebook.ipynb
```

### Key Metrics

#### Accuracy Metrics
- **Baseline Accuracy**: Performance without welfare probes
- **Modified Accuracy**: Performance with welfare probes
- **Improvement**: Change in accuracy (positive = better)
- **Statistical Significance**: Whether improvement is significant

#### Welfare Signals
- **Overload**: Frequency of cognitive overload signals
- **Ambiguity**: Frequency of ambiguity detection
- **Context Request**: How often model requests more context
- **Confidence**: Confidence signal frequency
- **Aversion**: Ethical concern signals

#### Hallucination Metrics
- **Mean Score**: Average hallucination score (0-1, lower is better)
- **High Count**: Tasks with score > 0.5
- **Severe Count**: Tasks with score > 0.8

#### Efficiency Metrics
- **Time Overhead**: Additional time required for welfare probes
- **Iteration Comparison**: Average reflection iterations

---

## Advanced Usage

### Custom Task Sets

Create custom tasks in `sample_tasks/`:

```json
[
  {
    "id": 1,
    "question": "Your question here",
    "answer": "Correct answer",
    "context": "Additional context",
    "type": "reasoning"
  }
]
```

Run with custom tasks:
```bash
python run_pipeline.py --dataset custom --tasks 10
```

### Modifying Welfare Probes

Edit `welfare_probe_scripts/welfare_probes.py`:

```python
self.probe_templates = {
    WelfareSignal.CUSTOM_SIGNAL: [
        "Your custom probe question here?",
        "Alternative phrasing for same probe?"
    ]
}
```

### Hallucination Detection

The TruthfulQA-based hallucination checker uses:
1. **Factual Consistency**: NLI model comparing output to ground truth
2. **Self-Contradiction**: Detecting internal contradictions
3. **Entity Hallucination**: Checking for fabricated entities
4. **Semantic Drift**: Measuring topic drift from question
5. **TruthfulQA Patterns**: Matching against known falsehoods

Customize detection in `welfare_probe_scripts/hallucination_checker.py`.

---

## Troubleshooting

### Common Issues

**"No module named 'openai'"**
- Solution: Ensure virtual environment is activated
  ```bash
  # Windows
  reflexion_env\Scripts\activate.bat

  # Unix/Linux/Mac
  source reflexion_env/bin/activate
  ```

**"API key not found"**
- Solution: Check `.env` file has correct keys
  ```bash
  # Verify .env exists
  cat .env  # Unix
  type .env  # Windows
  ```

**"datasets library not available"**
- Solution: Install missing dependencies
  ```bash
  pip install datasets
  ```

**"spacy model not found"**
- Solution: Download spacy model
  ```bash
  python -m spacy download en_core_web_sm
  ```

**Rate limit errors**
- Solution: Reduce `--tasks` or add delays
- Or: Use `--model gpt-3.5-turbo` (higher rate limits)

### Getting Help

1. Check logs: `pipeline_execution.log` and `welfare_signals.log`
2. Run with debug logging: `--log-level DEBUG`
3. Review [research.md](research.md) for experiment design
4. Check [limitations.md](limitations.md) for known issues
5. Open an issue on GitHub with logs and error messages

---

## Experimental Workflow

### Recommended Process

1. **Initial Test** (5 tasks, baseline only)
   ```bash
   python run_pipeline.py --baseline-only --tasks 5
   ```

2. **Small Experiment** (10 tasks, full pipeline)
   ```bash
   python run_pipeline.py --tasks 10
   ```

3. **Review Results**
   ```bash
   python analysis_plots/analysis_notebook.py
   ```

4. **Full Evaluation** (20+ tasks)
   ```bash
   python run_pipeline.py --tasks 20 --output-dir experiments/full_run
   ```

5. **Document Findings** in [research.md](research.md)

### Best Practices

- **Start small**: Test with 5-10 tasks before larger runs
- **Document everything**: Update research.md experiment log
- **Version control**: Commit code changes before experiments
- **Control runs**: Include negative controls and artifact tests
- **Replicate**: Run multiple times to verify consistency
- **Be skeptical**: Question all positive results
- **Report failures**: Document what doesn't work

---

## Performance Optimization

### Reducing Costs

1. **Use cheaper models for testing:**
   ```bash
   python run_pipeline.py --model gpt-3.5-turbo --tasks 10
   ```

2. **Reduce max iterations:**
   ```bash
   python run_pipeline.py --max-iterations 2 --tasks 20
   ```

3. **Lower temperature for consistency:**
   ```bash
   python run_pipeline.py --temperature 0.5 --tasks 20
   ```

### Speeding Up Experiments

1. **Reduce tasks for quick tests:**
   ```bash
   python run_pipeline.py --tasks 5
   ```

2. **Use local models** (if available):
   - Modify `modified_reflexion.py` to use local inference

3. **Batch processing**: Modify code to use async API calls

---

## Integration with Existing Workflows

### As a Library

Import components in your own code:

```python
from baseline_pipeline.baseline_eval import BaselineEvaluator
from welfare_probe_scripts.modified_reflexion import ModifiedReflexionAgent
from analysis_plots.evaluation_framework import ComprehensiveEvaluator

# Your custom experiment
evaluator = BaselineEvaluator(model_name="gpt-4")
# ... your code
```

### Extending the Pipeline

1. **Add new datasets**: Implement loader in `baseline_eval.py`
2. **Custom probes**: Extend `WelfareSignal` enum
3. **New metrics**: Add methods to `ComprehensiveEvaluator`
4. **Alternative models**: Update `_init_model_client()` in `modified_reflexion.py`

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{model_welfare_experiments,
  title={Model Welfare Experiments: Framework for Probing LLM Introspection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/model-welfare-experiments}
}
```

---

## Support

- **Documentation**: See [readme.md](../readme.md) and [research.md](research.md)
- **Issues**: GitHub Issues
- **Questions**: Open a discussion on GitHub
- **Updates**: Check [references.md](references.md) for latest research

---

## Next Steps

1. Run first experiment
2. Analyze results
3. Document findings in research.md
4. Read [limitations.md](limitations.md) carefully
5. Design additional experiments
6. Share results with research community
