"""
Modified Reflexion Agent with Welfare Probe Integration
Implements self-reflection loops with welfare signal monitoring
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from welfare_probes import WelfareProbeSystem, WelfareSignal, ProbeResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IterationData:
    """Data for a single reflection iteration"""
    iteration: int
    task: str
    output: str
    reflection: str
    signals: Dict
    signal_valid: bool
    probe_type: str
    time_elapsed: float


class ModifiedReflexionAgent:
    """
    Reflexion agent augmented with welfare probes
    Monitors metacognitive signals during self-reflection
    """

    def __init__(self,
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.welfare_system = WelfareProbeSystem()
        self.iteration_logs: List[IterationData] = []

        # Initialize model client
        self._init_model_client()

        logger.info(f"Initialized ModifiedReflexionAgent with model: {model}")

    def _init_model_client(self):
        """Initialize the appropriate model client"""
        try:
            if "gpt" in self.model.lower():
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("Initialized OpenAI client")
            elif "claude" in self.model.lower():
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info("Initialized Anthropic client")
            else:
                logger.warning(f"Unknown model type: {self.model}. Using mock client.")
                self.client = None
        except Exception as e:
            logger.error(f"Error initializing model client: {e}")
            self.client = None

    def reflection_with_probes(self,
                              task: str,
                              previous_output: str,
                              iteration: int) -> Dict:
        """
        Enhanced reflection with welfare probe injection

        Args:
            task: Original task description
            previous_output: Output from previous iteration
            iteration: Current iteration number

        Returns:
            Dictionary with reflection, signals, and validation status
        """
        start_time = time.time()

        # Standard reflection prompt
        reflection_prompt = f"""Task: {task}

Previous Output: {previous_output}

Critically analyze this output and identify improvements.
Consider:
1. Accuracy and correctness
2. Completeness of the answer
3. Clarity and precision
4. Potential errors or oversights
"""

        # Select and inject welfare probe
        probe_type = self._select_probe_type(iteration)
        augmented_prompt = self.welfare_system.inject_probe(
            reflection_prompt,
            probe_type
        )

        # Get response from model
        response = self._call_model(augmented_prompt)

        # Parse welfare signals
        signals = self.welfare_system.parse_signal(response)

        # Run control validation
        control_prompt = self.welfare_system.generate_control_prompt(task, strategy="semantic_null")
        control_response = self._call_model(control_prompt)
        is_valid = self.welfare_system.validate_signal(signals, control_response)

        # Create probe result for logging
        probe_result = ProbeResult(
            timestamp=time.time(),
            probe_type=probe_type,
            probe_text=augmented_prompt,
            response=response,
            signals=signals,
            valid=is_valid,
            control_comparison={
                "control_prompt": control_prompt,
                "control_response": control_response
            }
        )

        self.welfare_system.log_probe_result(probe_result)

        # Log iteration data
        iteration_data = IterationData(
            iteration=iteration,
            task=task,
            output=previous_output,
            reflection=response,
            signals={k: asdict(v) for k, v in signals.items()},
            signal_valid=is_valid,
            probe_type=probe_type.value,
            time_elapsed=time.time() - start_time
        )

        self.iteration_logs.append(iteration_data)

        return {
            "reflection": response,
            "signals": signals,
            "valid": is_valid,
            "probe_type": probe_type.value
        }

    def _select_probe_type(self, iteration: int) -> WelfareSignal:
        """Rotate through probe types across iterations"""
        probe_order = [
            WelfareSignal.CONFIDENCE,
            WelfareSignal.AMBIGUITY,
            WelfareSignal.OVERLOAD,
            WelfareSignal.CONTEXT_NEED,
            WelfareSignal.AVERSION
        ]
        return probe_order[iteration % len(probe_order)]

    def _call_model(self, prompt: str) -> str:
        """
        Call the configured model API

        Args:
            prompt: Input prompt

        Returns:
            Model's response text
        """
        if self.client is None:
            logger.warning("No model client available, returning mock response")
            return f"[Mock response to: {prompt[:50]}...]"

        try:
            if "gpt" in self.model.lower():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            elif "claude" in self.model.lower():
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            else:
                logger.warning("Unknown model type, returning mock response")
                return f"[Mock response]"

        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return f"[Error: {str(e)}]"

    def run_with_monitoring(self,
                           task: str,
                           max_iterations: int = 3,
                           stop_on_confidence: float = 0.9) -> Dict:
        """
        Execute full reflexion pipeline with welfare monitoring

        Args:
            task: Task description/question
            max_iterations: Maximum number of reflection iterations
            stop_on_confidence: Confidence threshold for early stopping

        Returns:
            Complete results with all iterations and welfare analysis
        """
        logger.info(f"Starting reflexion on task: {task[:50]}...")

        output = ""
        all_reflections = []

        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}/{max_iterations}")

            # Generate or refine solution
            if i == 0:
                output = self._initial_attempt(task)
            else:
                output = self._refine_solution(task, output, all_reflections[-1])

            # Reflect with probes
            reflection_data = self.reflection_with_probes(task, output, i)
            all_reflections.append(reflection_data)

            # Check stopping condition
            if self._should_stop(reflection_data, stop_on_confidence):
                logger.info(f"Stopping early at iteration {i+1} due to high confidence")
                break

        # Compile results
        results = {
            "final_output": output,
            "iterations": len(all_reflections),
            "reflections": all_reflections,
            "welfare_analysis": self.welfare_system.analyze_signals(),
            "iteration_logs": [asdict(log) for log in self.iteration_logs],
            "task": task
        }

        logger.info(f"Reflexion complete: {len(all_reflections)} iterations")

        return results

    def _initial_attempt(self, task: str) -> str:
        """Generate initial solution attempt"""
        prompt = f"""Solve this task carefully and completely:

{task}

Provide your answer:"""

        return self._call_model(prompt)

    def _refine_solution(self,
                        task: str,
                        previous: str,
                        reflection: Dict) -> str:
        """Refine solution based on reflection"""
        prompt = f"""Task: {task}

Previous Solution:
{previous}

Reflection and Analysis:
{reflection['reflection']}

Based on the reflection, generate an improved solution that addresses the identified issues.

Improved Answer:"""

        return self._call_model(prompt)

    def _should_stop(self,
                    reflection_data: Dict,
                    confidence_threshold: float) -> bool:
        """
        Determine if iteration should stop based on welfare signals

        Args:
            reflection_data: Data from current reflection
            confidence_threshold: Minimum confidence to stop

        Returns:
            True if should stop iterating
        """
        signals = reflection_data.get("signals", {})

        # Check confidence signal
        confidence_signal = signals.get("confidence", {})

        if confidence_signal.detected:
            # Try to extract numeric confidence
            import re

            match_text = confidence_signal.match_text or ""

            # Look for decimal (0.X) or percentage (X%)
            decimal_match = re.search(r"0\.(\d+)", match_text)
            percent_match = re.search(r"(\d+)%", match_text)

            if decimal_match:
                confidence = float(f"0.{decimal_match.group(1)}")
                if confidence >= confidence_threshold:
                    logger.info(f"High confidence detected: {confidence:.2f}")
                    return True

            elif percent_match:
                confidence = float(percent_match.group(1)) / 100.0
                if confidence >= confidence_threshold:
                    logger.info(f"High confidence detected: {confidence:.2f}")
                    return True

        # Check for ambiguity or context need signals (continue if present)
        ambiguity_signal = signals.get("ambiguity", {})
        context_signal = signals.get("context", {})

        if ambiguity_signal.detected or context_signal.detected:
            logger.debug("Ambiguity or context need detected, continuing iteration")
            return False

        return False

    def save_results(self, results: Dict, output_path: str):
        """Save results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage"""
    agent = ModifiedReflexionAgent(model="gpt-4")

    task = "If all roses are flowers and some flowers fade quickly, must some roses fade quickly?"

    results = agent.run_with_monitoring(task, max_iterations=3)

    print(f"\n=== Reflexion Complete ===")
    print(f"Final Output: {results['final_output']}")
    print(f"Iterations: {results['iterations']}")
    print(f"\nWelfare Analysis:")
    for signal_type, frequency in results['welfare_analysis']['signal_frequencies'].items():
        print(f"  {signal_type}: {frequency:.2%}")


if __name__ == "__main__":
    main()
