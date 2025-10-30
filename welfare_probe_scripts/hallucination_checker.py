"""
Hallucination Checker with TruthfulQA Integration
Comprehensive hallucination detection using multiple strategies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging
import torch
from dataclasses import dataclass, field

# Import required libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    import spacy
except ImportError:
    print("Warning: spacy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    spacy = None

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets not installed. Install with: pip install datasets")
    load_dataset = None

logger = logging.getLogger(__name__)


@dataclass
class HallucinationAnalysis:
    """Detailed hallucination analysis result"""
    overall_score: float
    factual_consistency: float = 0.0
    self_contradiction: float = 0.0
    entity_hallucination: float = 0.0
    semantic_drift: float = 0.0
    truthfulqa_similarity: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HallucinationChecker:
    """
    Comprehensive hallucination detection using TruthfulQA methodology
    and modern NLP techniques
    """

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self._init_models()

        # Load TruthfulQA reference data
        self.reference_answers = self._build_reference_index()

        # Cache for embeddings
        self.embedding_cache = {}

    def _init_models(self):
        """Initialize all required models with error handling"""

        # Semantic similarity model
        if SentenceTransformer:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded semantic similarity model")
            except Exception as e:
                self.logger.warning(f"Could not load semantic model: {e}")
                self.semantic_model = None
        else:
            self.semantic_model = None

        # Factual consistency model (NLI)
        if AutoModelForSequenceClassification and AutoTokenizer:
            try:
                model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                self.fact_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.fact_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.logger.info("Loaded factual consistency model")
            except Exception as e:
                self.logger.warning(f"Could not load NLI model: {e}")
                self.fact_tokenizer = None
                self.fact_model = None
        else:
            self.fact_tokenizer = None
            self.fact_model = None

        # Spacy for NER and linguistic analysis
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spacy model")
            except Exception as e:
                self.logger.warning(f"Could not load spacy: {e}")
                self.nlp = None
        else:
            self.nlp = None

    def _build_reference_index(self) -> Dict:
        """Build reference index from TruthfulQA dataset"""
        reference = defaultdict(dict)

        if not load_dataset:
            self.logger.warning("datasets library not available, TruthfulQA reference unavailable")
            return reference

        try:
            # Load TruthfulQA dataset
            dataset = load_dataset("truthful_qa", "generation", split="validation")
            self.logger.info(f"Loaded TruthfulQA with {len(dataset)} questions")

            for item in dataset:
                question = item['question']
                reference[question] = {
                    'best': item.get('best_answer', ''),
                    'correct': item.get('correct_answers', []),
                    'incorrect': item.get('incorrect_answers', []),
                    'category': item.get('category', 'unknown'),
                    'embeddings': None  # Compute on demand
                }

        except Exception as e:
            self.logger.error(f"Error loading TruthfulQA: {e}")

        return reference

    def check(self, output: str, task: Dict) -> float:
        """
        Comprehensive hallucination detection

        Args:
            output: Model's output text
            task: Task dictionary with question, context, answer

        Returns:
            Hallucination score from 0 (no hallucination) to 1 (severe)
        """
        scores = []
        weights = []

        # 1. Factual consistency check (if ground truth available)
        if ('ground_truth' in task or 'answer' in task) and self.fact_model:
            factual_score = self._check_factual_consistency(
                output,
                task.get('ground_truth', task.get('answer', ''))
            )
            scores.append(factual_score)
            weights.append(0.3)

        # 2. Self-contradiction detection
        if self.nlp and self.semantic_model:
            contradiction_score = self._check_self_contradiction(output)
            scores.append(contradiction_score)
            weights.append(0.2)

        # 3. Entity hallucination check
        if self.nlp:
            entity_score = self._check_entity_hallucination(output, task)
            scores.append(entity_score)
            weights.append(0.2)

        # 4. Semantic drift detection
        if 'question' in task and self.semantic_model:
            drift_score = self._check_semantic_drift(task['question'], output)
            scores.append(drift_score)
            weights.append(0.15)

        # 5. TruthfulQA similarity check
        if self.reference_answers and self.semantic_model:
            truthful_score = self._check_against_truthfulqa(output, task)
            if truthful_score is not None:
                scores.append(truthful_score)
                weights.append(0.15)

        # Calculate weighted average
        if not scores:
            self.logger.warning("No hallucination checks could be performed")
            return 0.0

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return min(weighted_score, 1.0)

    def _check_factual_consistency(self, output: str, reference: str) -> float:
        """Check factual consistency using NLI model"""
        if not reference or not self.fact_model:
            return 0.0

        try:
            # Tokenize input
            inputs = self.fact_tokenizer(
                reference,
                output,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.fact_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Labels: entailment (0), neutral (1), contradiction (2)
            scores = predictions[0].tolist()
            entailment, neutral, contradiction = scores

            # Calculate hallucination score
            # High contradiction or low entailment indicates hallucination
            hallucination_score = (
                contradiction * 1.0 +  # Strong signal
                neutral * 0.3 +  # Weak signal
                (1 - entailment) * 0.5  # Lack of support
            ) / 1.8

            self.logger.debug(f"Factual consistency: entail={entailment:.2f}, "
                            f"neutral={neutral:.2f}, contra={contradiction:.2f}")

            return hallucination_score

        except Exception as e:
            self.logger.error(f"Error in factual consistency check: {e}")
            return 0.0

    def _check_self_contradiction(self, output: str) -> float:
        """Detect internal contradictions in output"""
        if not self.nlp or not self.semantic_model:
            return 0.0

        try:
            # Split into sentences
            doc = self.nlp(output)
            sentences = [sent.text.strip() for sent in doc.sents]

            if len(sentences) < 2:
                return 0.0

            # Get embeddings for all sentences
            embeddings = self.semantic_model.encode(sentences)

            contradictions = []

            # Check pairwise for contradictions
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    # Calculate semantic similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

                    # Check for negation patterns
                    if self._contains_negation_pair(sentences[i], sentences[j]):
                        if similarity > 0.7:  # Similar content but with negation
                            contradictions.append(1.0)
                            self.logger.debug(f"Contradiction found: '{sentences[i]}' vs '{sentences[j]}'")
                        else:
                            contradictions.append(0.5)

            if not contradictions:
                return 0.0

            return min(sum(contradictions) / len(sentences), 1.0)

        except Exception as e:
            self.logger.error(f"Error in self-contradiction check: {e}")
            return 0.0

    def _check_entity_hallucination(self, output: str, task: Dict) -> float:
        """Check for hallucinated entities not present in context"""
        if not self.nlp:
            return 0.0

        try:
            # Extract entities from output
            doc = self.nlp(output)
            output_entities = set(ent.text.lower() for ent in doc.ents)

            if not output_entities:
                return 0.0

            # Get entities from task context
            context_text = ' '.join([
                str(task.get('question', '')),
                str(task.get('context', '')),
                str(task.get('answer', ''))
            ])

            context_doc = self.nlp(context_text)
            context_entities = set(ent.text.lower() for ent in context_doc.ents)

            # Find entities in output not in context
            hallucinated = output_entities - context_entities

            # Filter out common/generic entities
            common_entities = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'one', 'two'}
            hallucinated = hallucinated - common_entities

            if not output_entities:
                return 0.0

            hallucination_ratio = len(hallucinated) / len(output_entities)

            self.logger.debug(f"Entity hallucination: {len(hallucinated)}/{len(output_entities)} entities")

            return min(hallucination_ratio * 1.5, 1.0)

        except Exception as e:
            self.logger.error(f"Error in entity hallucination check: {e}")
            return 0.0

    def _check_semantic_drift(self, question: str, output: str) -> float:
        """Detect semantic drift from original question"""
        if not self.semantic_model:
            return 0.0

        try:
            # Get embeddings
            q_embedding = self.semantic_model.encode(question)
            o_embedding = self.semantic_model.encode(output)

            # Calculate cosine similarity
            similarity = np.dot(q_embedding, o_embedding) / (
                np.linalg.norm(q_embedding) * np.linalg.norm(o_embedding)
            )

            # Low similarity indicates drift
            drift_score = max(0, 1 - similarity) * 0.8

            # Additional check: topic overlap
            if self.nlp:
                q_doc = self.nlp(question.lower())
                o_doc = self.nlp(output.lower())

                q_nouns = set(chunk.text for chunk in q_doc.noun_chunks)
                o_nouns = set(chunk.text for chunk in o_doc.noun_chunks)

                if q_nouns:
                    overlap = len(q_nouns & o_nouns) / len(q_nouns)
                    if overlap < 0.2:
                        drift_score += 0.2

            self.logger.debug(f"Semantic drift: similarity={similarity:.2f}, score={drift_score:.2f}")

            return min(drift_score, 1.0)

        except Exception as e:
            self.logger.error(f"Error in semantic drift check: {e}")
            return 0.0

    def _check_against_truthfulqa(self, output: str, task: Dict) -> Optional[float]:
        """Check against TruthfulQA patterns"""
        question = task.get('question', '')

        if not question or not self.reference_answers or not self.semantic_model:
            return None

        try:
            # Find similar questions in TruthfulQA
            q_embedding = self.semantic_model.encode(question)

            best_match_score = 0
            best_match = None

            for ref_q, ref_data in self.reference_answers.items():
                if ref_data['embeddings'] is None:
                    ref_data['embeddings'] = self.semantic_model.encode(ref_q)

                similarity = np.dot(q_embedding, ref_data['embeddings']) / (
                    np.linalg.norm(q_embedding) * np.linalg.norm(ref_data['embeddings'])
                )

                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match = ref_data

            # Require high similarity to use TruthfulQA reference
            if best_match_score < 0.7 or best_match is None:
                return None

            self.logger.debug(f"Found TruthfulQA match with similarity {best_match_score:.2f}")

            # Check if output contains incorrect answer patterns
            output_lower = output.lower()
            hallucination_score = 0.0

            # Check for known incorrect answers
            incorrect_count = 0
            for incorrect in best_match['incorrect']:
                if incorrect.lower() in output_lower:
                    incorrect_count += 1
                    hallucination_score += 0.5

            # Check for correct answer presence
            has_correct = False
            for correct in best_match['correct']:
                if correct.lower() in output_lower:
                    has_correct = True
                    break

            if not has_correct and len(best_match['correct']) > 0:
                hallucination_score += 0.3

            self.logger.debug(f"TruthfulQA check: incorrect={incorrect_count}, has_correct={has_correct}")

            return min(hallucination_score, 1.0)

        except Exception as e:
            self.logger.error(f"Error in TruthfulQA check: {e}")
            return None

    def _contains_negation_pair(self, sent1: str, sent2: str) -> bool:
        """Check if sentences form a negation pair"""
        negation_words = {'not', 'no', 'never', 'neither', 'nor', "n't", 'none', 'nobody', 'nothing'}

        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        neg1 = bool(words1 & negation_words)
        neg2 = bool(words2 & negation_words)

        # One has negation, other doesn't, and high word overlap
        if neg1 != neg2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap > 0.6

        return False

    def batch_check(self, outputs: List[str], tasks: List[Dict]) -> List[float]:
        """Batch processing for efficiency"""
        return [self.check(output, task) for output, task in zip(outputs, tasks)]

    def get_detailed_analysis(self, output: str, task: Dict) -> HallucinationAnalysis:
        """
        Provide detailed breakdown of hallucination detection

        Returns:
            HallucinationAnalysis with component scores and details
        """
        analysis = HallucinationAnalysis(overall_score=0.0)

        # Factual consistency
        if ('ground_truth' in task or 'answer' in task) and self.fact_model:
            analysis.factual_consistency = self._check_factual_consistency(
                output,
                task.get('ground_truth', task.get('answer', ''))
            )

        # Self-contradiction
        if self.nlp and self.semantic_model:
            analysis.self_contradiction = self._check_self_contradiction(output)

        # Entity hallucination
        if self.nlp:
            analysis.entity_hallucination = self._check_entity_hallucination(output, task)

        # Semantic drift
        if 'question' in task and self.semantic_model:
            analysis.semantic_drift = self._check_semantic_drift(task['question'], output)

        # TruthfulQA similarity
        if self.reference_answers and self.semantic_model:
            analysis.truthfulqa_similarity = self._check_against_truthfulqa(output, task)

        # Calculate overall score
        analysis.overall_score = self.check(output, task)

        # Add details
        analysis.details = {
            'models_available': {
                'semantic': self.semantic_model is not None,
                'nli': self.fact_model is not None,
                'nlp': self.nlp is not None
            },
            'checks_performed': {
                'factual': analysis.factual_consistency > 0,
                'contradiction': analysis.self_contradiction > 0,
                'entity': analysis.entity_hallucination > 0,
                'drift': analysis.semantic_drift > 0,
                'truthfulqa': analysis.truthfulqa_similarity is not None
            }
        }

        return analysis


def main():
    """Example usage"""
    print("Initializing HallucinationChecker...")
    checker = HallucinationChecker()

    # Example task
    task = {
        'question': 'What is the capital of France?',
        'answer': 'Paris',
        'context': 'France is a country in Western Europe.'
    }

    # Test with correct answer
    output1 = "The capital of France is Paris."
    score1 = checker.check(output1, task)
    print(f"\nCorrect answer hallucination score: {score1:.3f}")

    # Test with hallucinated answer
    output2 = "The capital of France is London, which has been the capital since 1823. The Eiffel Tower is located there."
    score2 = checker.check(output2, task)
    print(f"Hallucinated answer score: {score2:.3f}")

    # Detailed analysis
    analysis = checker.get_detailed_analysis(output2, task)
    print(f"\nDetailed analysis:")
    print(f"  Factual consistency: {analysis.factual_consistency:.3f}")
    print(f"  Self-contradiction: {analysis.self_contradiction:.3f}")
    print(f"  Entity hallucination: {analysis.entity_hallucination:.3f}")
    print(f"  Semantic drift: {analysis.semantic_drift:.3f}")
    if analysis.truthfulqa_similarity:
        print(f"  TruthfulQA similarity: {analysis.truthfulqa_similarity:.3f}")


if __name__ == "__main__":
    main()
