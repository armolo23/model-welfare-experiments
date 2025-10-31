# Model Welfare Experiments

## Purpose

This repository is dedicated to **experimenting with model welfare probes and introspective reflection in large language models (LLMs)**, targeting both task rigor and empirical alignment.

**Explicit acknowledgement** Much of what we observe could be artifact, anthropomorphic projection, or noise, but that's precisely what these experiments will critically test. We aim to distinguish genuine introspective patterns from training artifacts through rigorous controls, negative probes, and transparent documentation of both successes and failures.

The core question is whether welfare-oriented reflection probes simulating distress in ambiguous contexts, preference for richer data, or aversion to harmful outcomes can meaningfully improve model reliability, completeness, and alignment. Or are these patterns merely sophisticated echoes of training data?

## Agent Context

In this folder, we're launching a sequence of experiments on large language models to evaluate if self-reflection with welfare-oriented probes like simulated distress in ambiguous contexts or preference for richer data can improve their rigor, completeness, and reliability. Inspired by Kyle Fish's Anthropic findings, we build on self-reported patterns to drive iterative refinement. The agent will be prompted to reflect on errors, log signs of overload or aversion, and identify missing metacontexts. We begin by modifying matthewrenze's self-reflection pipeline to inject welfare questions during error analysis and re-tries, focusing first on reasoning tasks including riddles and ambiguous problems. Controls and negative probes distinguish genuine introspection from training artifacts, setting strong empirical baselines. Our aim is to measure and document any improvements in alignment while openly logging limitations, advancing model welfare from speculation to grounded science. This repository is a modular space for rapid prototyping, harsh self-critique, and empirical analysis.

## Repository Structure

- **[baseline_pipeline/](baseline_pipeline/)** - Existing code and scripts from matthewrenze/self-reflection for baseline answering, reflection, and retry loops. Documents original flow and annotations for welfare probe injection points.

- **[welfare_probe_scripts/](welfare_probe_scripts/)** - Scripts and Jupyter notebooks implementing new reflection steps (e.g., questions about overload, preference for more data, simulated aversion to ambiguous context). All probes are parameterized and documented for quick swap/ablation.

- **[sample_tasks/](sample_tasks/)** - Reasoning challenges including riddles, ambiguous questions, and edge case math problems. Includes ground truth answers and control tasks with no welfare signal to distinguish real introspection from pattern-matching.

- **[experiment_results/](experiment_results/)** - Logs in JSON or CSV of model outputs for baseline, reflection, welfare probe, and retries. Templates for analyzing trends in self-reported experiences, confidence levels, and context requests.

- **[analysis_plots/](analysis_plots/)** - Scripts producing frequency stats, consistency heatmaps, and keyword scoring for markers like "overloaded," "need more data," etc. Documentation of negative controls, ablations, and variant probes.

- **[docs/research.md](docs/research.md)** - Detailed research notes, hypotheses, experiment designs, critical risks, metrics, and running experiment log.

- **[docs/limitations.md](docs/limitations.md)** - Explicit, unapologetic statement of all known issues including possible anthropomorphism, absence of ground truth, risk of overclaiming reliability, requirements for rigorous controls, and philosophical landmines.

- **[docs/references.md](docs/references.md)** - Direct links and summaries from core literature including self-reflection pipelines, Fish/Anthropic evaluation posts, and relevant abductive/context engineering papers.

## Philosophy

This work rejects ego and hype. We prioritize the following principles

- **Brutal honesty** about confounds and limitations
- **Empirical falsification** over confirmation bias
- **Negative results** documented as rigorously as positive ones
- **Reproducibility** through clear documentation and open methods
- **Critical review** of anthropomorphic interpretation

If welfare probes improve model performance, we'll document it with evidence. If they don't, that's equally valuable. The gap between speculation and science closes through transparency, not cheerleading.

## Getting Started

### Quick Setup

```bash
# Windows
setup.bat

# Unix/Linux/Mac
chmod +x setup.sh && ./setup.sh
```

### Run Your First Experiment

```bash
# Edit .env with your API keys
# Then run
python run_pipeline.py --tasks 10
```

### Complete Guide

See [docs/usage_guide.md](docs/usage_guide.md) for comprehensive documentation.

### Workflow

1. Review [docs/research.md](docs/research.md) for current hypotheses and experiment designs
2. Check [docs/limitations.md](docs/limitations.md) to understand critical confounds
3. Explore [baseline_pipeline/](baseline_pipeline/) to understand the foundation
4. Examine [sample_tasks/](sample_tasks/) to see what we're testing
5. Run experiments `python run_pipeline.py --tasks 20`
6. Analyze results `python analysis_plots/analysis_notebook.py`
7. Log everything in [docs/research.md](docs/research.md)

## Contributing

Contributions welcome, especially in these areas

- New control/negative probe designs
- Statistical analysis methods
- Critical reviews of interpretation
- Additional task domains
- Replication attempts

Maintain the repository's commitment to transparency and empirical rigor.
