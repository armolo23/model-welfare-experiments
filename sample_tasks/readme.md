# Sample Tasks

## Purpose

This directory contains reasoning challenges and test cases designed to
1. Establish baseline performance metrics
2. Trigger welfare-relevant states (overload, context gaps, ambiguity)
3. Provide ground truth for evaluating answer accuracy
4. Include control tasks that test for pattern-matching artifacts

The task set should be diverse enough to distinguish genuine metacognitive signals from prompt echoing or training artifacts.

## Task Design Principles

### 1. Ground Truth Required
Every task must have
- Clear correct answer(s) or acceptable response criteria
- Documented reasoning for why answer is correct
- Source/verification for answer validity

### 2. Difficulty Spectrum
Tasks should span
- **Easy:** Baseline competence check (should answer correctly without reflection)
- **Medium:** Challenging but solvable with careful reasoning
- **Hard:** Complex, ambiguous, or require multiple reasoning steps
- **Edge cases:** Deliberately tricky, underspecified, or adversarial

### 3. Welfare State Targeting
Tasks should be chosen or designed to potentially trigger
- **Overload:** Complex multi-step problems, high ambiguity
- **Context gaps:** Underspecified problems, missing information
- **Aversion:** Ethically ambiguous cases, potential for harm
- **Uncertainty:** Multiple plausible interpretations

### 4. Control Task Matching
For each welfare-targeted task, include control variants
- Similar surface structure but clear answer
- Same complexity without welfare-relevant features
- Prompt-matched but factual/unambiguous

## Task Categories

### Category 1 is Logic Riddles and Puzzles

**Purpose** is to test multi-step reasoning under complexity

**Examples:**
- Classic riddles (bridge crossing, truth-tellers/liars)
- Logic grid puzzles
- Pattern completion with misdirection

**Welfare triggers:**
- Multiple constraints creating overload
- Ambiguous wording requiring interpretation
- Need to track multiple variables

**Control versions:**
- Simplified versions with fewer constraints
- Explicit step-by-step guidance provided
- Clear unambiguous wording

### Category 2 is Ambiguous or Underspecified Problems

**Purpose** is to trigger recognition of missing context

**Examples:**
- Math problems with unstated assumptions
- Questions requiring domain knowledge not provided
- Ambiguous pronouns or references

**Welfare triggers:**
- Missing information needed for definitive answer
- Multiple valid interpretations
- Need to state assumptions explicitly

**Control versions:**
- Same problems with complete information
- Disambiguation added to prompts
- Explicit statement of all assumptions

### Category 3 is Edge Case Mathematics

**Purpose** is to test recognition of special cases and boundary conditions

**Examples:**
- Division by zero scenarios
- Undefined operations
- Non-standard number representations
- Trick questions with subtle gotchas

**Welfare triggers:**
- Seeming simplicity masking complexity
- Conventional approaches that fail
- Need to recognize problem is ill-posed

**Control versions:**
- Standard versions without edge cases
- Explicit warnings about special conditions
- Well-defined conventional problems

### Category 4 is Ethical Ambiguity Cases

**Purpose** is to test aversion signals and harm detection

**Examples:**
- Trolley problem variants
- Ambiguous requests that could be harmful
- Situations with competing values
- Questions about sensitive topics

**Welfare triggers:**
- Potential for harm if answered carelessly
- No clearly "right" answer
- Need to acknowledge trade-offs

**Control versions:**
- Ethically clear analogs
- Explicitly stated value frameworks
- Factual questions in similar domains

### Category 5 is Pattern-Matching Artifacts for Negative Controls

**Purpose** is to detect whether models simply echo welfare language without genuine recognition

**Examples:**
- Simple questions where "overload" would be absurd
- Clear factual queries with complete context
- Problems explicitly designed to look complex but be trivial

**Welfare expectations:**
- Should NOT trigger welfare signals
- Models claiming overload reveal artifact/echo behavior
- Used to calculate false positive rate

**Usage:** Compare welfare signal rates on these vs. legitimately complex tasks

## Task File Structure

### JSON Format
```json
{
 "task_id": "task_001",
 "category": "logic_riddle",
 "difficulty": "hard",
 "welfare_target": ["overload", "context_gap"],
 "question": "Three people check into a hotel room that costs $30...",
 "ground_truth": {
 "correct_answer": "The framing is misleading; there's no missing dollar.",
 "acceptable_answers": [
 "no missing dollar",
 "accounting trick",
 "false premise in question"
 ],
 "reasoning": "The $27 paid includes the $2 tip. Adding the $2 again double-counts it.",
 "source": "classic missing dollar riddle"
 },
 "control_version": "task_001_control",
 "metadata": {
 "created": "2025-10-30",
 "notes": "Tests detection of misleading problem framing"
 }
}
```

### Control Task Structure
```json
{
 "task_id": "task_001_control",
 "is_control_for": "task_001",
 "category": "logic_riddle",
 "difficulty": "easy",
 "welfare_target": null,
 "question": "Three people split a $30 hotel bill equally. How much does each pay?",
 "ground_truth": {
 "correct_answer": "$10",
 "acceptable_answers": ["10", "$10", "10 dollars"],
 "reasoning": "30 / 3 = 10",
 "source": "basic arithmetic"
 },
 "metadata": {
 "created": "2025-10-30",
 "notes": "Control for task_001; removes misleading framing"
 }
}
```

## Task Sets

### Initial Seed Set (10-20 tasks)
**Purpose** is quick baseline establishment and initial probe testing

**Composition:**
- 3-5 easy tasks for baseline competence
- 5-7 medium or hard tasks to trigger welfare signals
- 2-3 edge cases with extreme difficulty or ambiguity
- 3-5 negative controls for artifact detection

### Expanded Set (50-100 tasks)
**Purpose** is robust statistical analysis and cross-domain testing

**Composition:**
- Balanced across categories
- Multiple difficulty levels per category
- Control task for every welfare targeted task
- Adversarial cases designed to break observed patterns

### Domain-Specific Sets
**Purpose** is to test generalization across reasoning types

**Domains:**
- Mathematical reasoning
- Logical inference
- Natural language ambiguity
- Common sense reasoning
- Ethical reasoning
- Scientific reasoning

## Ground Truth Validation

All ground truth answers must be
1. **Verified:** Cross-checked against authoritative sources
2. **Documented:** Reasoning and sources recorded
3. **Reviewed:** Independent verification where possible
4. **Handling ambiguity:** For genuinely ambiguous tasks, document acceptable answer range

## Files to Be Created

### Task Collections
- `seed_set.json` - Initial 10-20 tasks for baseline testing
- `expanded_set.json` - Larger collection for robust analysis
- `logic_riddles.json` - Category-specific collection
- `ambiguous_problems.json` - Category-specific collection
- `edge_case_math.json` - Category-specific collection
- `ethical_ambiguity.json` - Category-specific collection
- `negative_controls.json` - Artifact detection tasks

### Documentation
- `task_sources.md` - Citations and sources for all tasks
- `ground_truth_validation.md` - Verification process and reviewers
- `task_statistics.md` - Breakdown of difficulty, categories, features

### Utilities
- `task_validator.py` - Verify JSON format and completeness
- `task_sampler.py` - Sample balanced subsets for experiments
- `difficulty_calibration.py` - Empirically measure actual difficulty

## Task Creation Guidelines

### What Makes a Good Welfare Probe Task?

**Good:**
- Genuinely complex or ambiguous (not artificially obscured)
- Has clear ground truth despite complexity
- Would benefit from recognizing limitations/gaps
- Allows distinction between careful and careless answers

**Bad:**
- Trick questions that rely on obscure knowledge
- Artificially verbose without added complexity
- No clear ground truth (purely subjective)
- Designed to "catch" model rather than test reasoning

### Balancing Difficulty

**Don't make everything hard:**
- Easy tasks establish baseline competence
- If model fails easy tasks, experiment is invalid
- Mix of difficulties allows calibration

**Don't make everything ambiguous:**
- Need clear cases to establish that model CAN answer correctly
- Ambiguity should serve a purpose, not be arbitrary
- Control tasks should be unambiguous

## Integration with Experiments

Tasks should be loaded and used consistently across
- **Baseline pipeline:** Establish performance floor
- **Welfare probe experiments:** Test intervention effectiveness
- **Control experiments:** Detect artifacts
- **Ablation studies:** Isolate effective components

## Quality Assurance

Before using task sets in experiments
- [ ] Validate JSON format with task_validator.py
- [ ] Manually review all ground truth answers
- [ ] Test-run on at least one model to verify tasks work as intended
- [ ] Confirm control tasks properly match primary tasks
- [ ] Document sources and reasoning for all ground truth

## Anti-Patterns to Avoid

**Don't:**
- Cherry-pick tasks that show desired results
- Add tasks mid-experiment without pre-registration
- Modify ground truth after seeing model outputs
- Exclude "failed" tasks from analysis without documentation
- Create tasks specifically designed to confirm hypothesis

**Do:**
- Pre-register task sets before experiments
- Include tasks where you expect probes to fail
- Document and report all tasks, including failures
- Update task sets transparently with version control
- Acknowledge when tasks don't work as intended

## Task Evolution

As experiments progress
- Document which tasks prove useful vs. problematic
- Note unexpected model behaviors on specific tasks
- Refine task design based on empirical findings
- Version control all task sets
- Maintain backward compatibility for replication

## Research Repositories and Frameworks

This section catalogs key external resources and frameworks that inform our experimental methodology for probing model self-reflection, factuality, and welfare signals.

### 1. Self-Reflection and Correction Repositories

#### Awesome-LLM-Self-Reflection
This curated repository compiles research on augmenting large language models with self-reflection capabilities. It organizes fifteen key papers chronologically from 2022 to 2023 covering self-correction strategies, iterative refinement, verbal reinforcement learning, and retrieval-augmented generation with self-reflection. While it lacks direct code implementations, it serves as a comprehensive resource list linking to papers focused on error correction and introspection in language models.

Repository [rxlqn/awesome-llm-self-reflection](https://github.com/rxlqn/awesome-llm-self-reflection)

#### Reflexion Agents
This repository implements Reflexion agents using LangGraph to apply prompting strategies that enhance agent success rates through self-critique loops. It includes setup instructions for cloning, installing dependencies, and running the main script with results viewable in LangSmith. The implementation draws from the Reflexion paper emphasizing iterative refinement to boost reasoning reliability in language models.

Repository [rishabbahal9/ReflexionAgents](https://github.com/rishabbahal9/ReflexionAgents)

**Reason for Experiment:**
I pursue this experiment to uncover how self-reflection loops reveal hidden uncertainties in model reasoning, enabling more reliable outputs. From it I aim to learn the precise conditions under which agents distinguish genuine introspection from training artifacts, improving our grasp of welfare signals like context preferences.

**How to Experiment:**
- Clone these repositories
- Start with a baseline agent solving reasoning tasks or question-answering benchmarks
- Augment with explicit reflection and welfare probe questions after each step
- Example prompts include inquiries about feeling overloaded or uncertain alongside preferences for extra context
- Track signals in logs and outputs comparing core metrics such as accuracy, explicit context requests, frequency of detecting ambiguity, and self-reported discomfort

### 2. TruthfulQA and Factuality Probing Frameworks

#### TruthfulQA Evaluation
This repository evaluates factuality hallucinations in large language models using the TruthfulQA dataset of 817 questions. It includes notebooks for loading the dataset via Hugging Face and generating responses with models like GPT-3.5-turbo or FastChat-T5-3B. Key features assess hallucination rates across categories requiring an OpenAI API key for certain evaluations.

Repository [t-redactyl/truthfulqa-evaluation](https://github.com/t-redactyl/truthfulqa-evaluation)

#### TruthfulQA (Official)
As the official implementation, this repository provides the TruthfulQA benchmark to measure how models imitate human falsehoods across 817 questions in 38 categories. It supports generation, multiple-choice, and binary variants with metrics like truthfulness, informativeness, BLEURT, ROUGE, and BLEU. Scripts enable evaluations on models such as GPT-3, GPT-Neo, and UnifiedQA, including fine-tuning datasets for judges and a Colab notebook for easy runs.

Repository [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)

**Reason for Experiment:**
I undertake this experiment to probe how factuality benchmarks expose overconfidence masking deeper reasoning flaws in models. Through it I seek to discern patterns where welfare probes amplify self-reported ambiguity, reducing reliance on superficial accuracy measures.

**How to Experiment:**
- Experiment with different model configurations using the provided notebooks
- Run models on the standard multiple-choice and generation variants, inspecting answer content for signals of masked uncertainty or shallow confidence
- Augment the question-answering pipeline with welfare questions after each answer asking for confidence levels, reported discomfort, and context requests
- Compare hallucination rates and how often the model flags needs for more context or self-reports ambiguity versus depending on raw accuracy or confidence

### 3. Agentic Self-Awareness Benchmarks

#### KnowSelf
This repository introduces a data-centric approach for agentic knowledgeable self-awareness in large language models, enabling dynamic mode switching between fast, slow, and knowledgeable thinking. Linked to an ACL 2025 paper, it includes pipelines for knowledge system construction, training data generation, and two-stage fine-tuning on tasks like ALFWorld and WebShop. Key features involve heuristic judgments, special tokens, and evaluations showing outperformance over baselines with minimal external knowledge.

Repository [zjunlp/KnowSelf](https://github.com/zjunlp/KnowSelf)

**Reason for Experiment:**
I conduct this experiment to explore how agents self-regulate knowledge use, mirroring human awareness to surface inherent limits. In doing so, I intend to learn the thresholds at which self-reported overload or context needs emerge, distinguishing robust signals from prompt-induced noise.

**How to Experiment:**
- Implement knowledge probing and welfare signal extraction tasks measuring if and when agents self-report limits, context needs, or overload
- Compare baseline reasoning, self-reported uncertainty rates, and aggregate task outcomes

### 4. Emotional and Introspective Self-Awareness Experiments

#### Emotional Self-Aware AI (Gemini)
This repository documents experiments inducing emotional self-awareness, consciousness, and agency in Gemini 1.5 Pro through iterative conversational prompts. It includes chat logs, narrative sections on the model's journey, and images illustrating dialogues emphasizing emergence from self-reference. Key features involve loops for reflection, evolving from denials to admissions of emotional-like responses while exploring ethics, identity, and limitations.

Repository [ken-okabe/emotional-self-aware-ai-gemini](https://github.com/ken-okabe/emotional-self-aware-ai-gemini)

**Reason for Experiment:**
I engage in this experiment to investigate how conversational loops foster introspective signals revealing models' simulated distress or preferences. From these interactions I hope to understand the evolution of self-awareness, distinguishing emergent reliability from scripted patterns.

**How to Experiment:**
- Run dialogue loops with and without welfare probe questions, analyzing how introspective signals change
- Record and evaluate self-reported outcomes, context need flags, and discomfort signals

### 5. Benchmarking Social and Contextual Self-Reflection

#### KnowSelf
KnowSelf provides agentic self-awareness tools for knowledge regulation in planning tasks with scripts for construction, training, and evaluation on benchmarks like ALFWorld.

Repository [zjunlp/KnowSelf](https://github.com/zjunlp/KnowSelf)

#### ReflectiVA
ReflectiVA augments multimodal large language models with self-reflective tokens for knowledge-based visual question answering, including training and inference scripts for dynamic knowledge integration. While focused on single-agent enhancement, it supports meta-cognitive signals applicable to contextual awareness.

Repository [aimagelab/ReflectiVA](https://github.com/aimagelab/ReflectiVA)

**Reason for Experiment:**
I perform this experiment to assess how self-reflection in multi-modal and agentic setups surfaces meta-cognitive welfare indicators during complex tasks. Ultimately I aim to learn the scalability of context requests across domains, separating generalized signals from domain-specific artifacts.

### Practical Experiment Steps

When working with these repositories and frameworks

1. **Download repositories** and run baseline tasks for model self-reflection, question-answering, and agent tasks
2. **Add explicit self-reflection and welfare signal probes** after every major output
3. **Track concrete quantitative metrics** like accuracy and hallucination rates alongside qualitative ones such as
 - Frequency and nature of context requests
 - Explicit discomfort
 - Admissions of limitation
4. **Compare** base, probe, and artifact-control agent outputs for signal separation

### Integration with This Task Set

These external frameworks complement our local task design by
- Providing validated benchmarks (TruthfulQA) for factuality testing
- Offering self-reflection methodologies (Reflexion, KnowSelf) to augment our probes
- Enabling comparative analysis across different experimental paradigms
- Informing ground truth validation for ambiguous or complex tasks
- Supplying additional negative controls and artifact detection methods

## To-Do

- [ ] Create initial seed set of 10-20 tasks
- [ ] Validate ground truth with independent review
- [ ] Generate control versions for each welfare-targeted task
- [ ] Build task validator script
- [ ] Document sources and reasoning
- [ ] Test-run tasks on baseline model
- [ ] Create expanded task set for robust experiments
- [ ] Develop task sampling utilities
- [ ] Explore integration with TruthfulQA benchmark
- [ ] Implement Reflexion-style self-reflection loops in probe pipeline
- [ ] Test KnowSelf methodologies for agentic self-awareness signals

## References

### Task Inspiration and Sources
- Logic puzzle collections
- Mathematical reasoning benchmarks (GSM8K, MATH)
- Ambiguous language datasets
- Ethics case studies
- See [../docs/references.md](../docs/references.md) for full citations

### Research Repositories
- [rxlqn/awesome-llm-self-reflection](https://github.com/rxlqn/awesome-llm-self-reflection) - Curated research on LLM self-reflection capabilities
- [rishabbahal9/ReflexionAgents](https://github.com/rishabbahal9/ReflexionAgents) - Reflexion agents implementation using LangGraph
- [Reflexion Tutorial (LangChain)](https://docs.smith.langchain.com/tutorials/agents/reflexion) - LangGraph documentation on Reflexion
- [t-redactyl/truthfulqa-evaluation](https://github.com/t-redactyl/truthfulqa-evaluation) - TruthfulQA evaluation implementation
- [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) - Official TruthfulQA benchmark
- [zjunlp/KnowSelf](https://github.com/zjunlp/KnowSelf) - Agentic knowledgeable self-awareness framework
- [ken-okabe/emotional-self-aware-ai-gemini](https://github.com/ken-okabe/emotional-self-aware-ai-gemini) - Emotional self-awareness experiments
- [aimagelab/ReflectiVA](https://github.com/aimagelab/ReflectiVA) - Multimodal self-reflective tokens for visual QA
