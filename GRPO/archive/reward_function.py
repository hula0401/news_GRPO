"""
Reward Functions for GRPO Training

This module provides custom reward functions for evaluating model outputs
during GRPO training. It supports both GSM8K math problems and can be
extended for news data evaluation.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Result of reward calculation"""
    score: float
    details: Dict[str, Any]


class GSM8KRewardFunction:
    """
    Reward function for GSM8K math problems.

    Evaluates model responses by:
    1. Extracting the final numerical answer
    2. Comparing with ground truth
    3. Awarding partial credit for reasoning steps
    """

    def __init__(
        self,
        correct_answer_score: float = 1.0,
        wrong_answer_score: float = 0.0,
        partial_credit: bool = True,
        reasoning_weight: float = 0.2,
    ):
        """
        Initialize GSM8K reward function.

        Args:
            correct_answer_score: Score for correct final answer
            wrong_answer_score: Score for incorrect final answer
            partial_credit: Whether to award partial credit for reasoning
            reasoning_weight: Weight for reasoning quality (0-1)
        """
        self.correct_answer_score = correct_answer_score
        self.wrong_answer_score = wrong_answer_score
        self.partial_credit = partial_credit
        self.reasoning_weight = reasoning_weight

    def extract_answer(self, text: str) -> Optional[float]:
        """
        Extract numerical answer from model output.

        Looks for patterns like:
        - "#### 42"
        - "The answer is 42"
        - "Answer: 42"

        Args:
            text: Model generated text

        Returns:
            Extracted number or None if not found
        """
        # Pattern 1: GSM8K format "#### number"
        pattern1 = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(pattern1, text)
        if match:
            return float(match.group(1).replace(',', ''))

        # Pattern 2: "The answer is number"
        pattern2 = r'(?:the answer is|answer:|final answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))

        # Pattern 3: Last number in text (fallback)
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return float(numbers[-1].replace(',', ''))

        return None

    def extract_ground_truth(self, ground_truth_text: str) -> Optional[float]:
        """
        Extract numerical answer from ground truth.

        Args:
            ground_truth_text: Ground truth answer string

        Returns:
            Extracted number or None
        """
        # GSM8K ground truth format: "explanation\n#### answer"
        pattern = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(pattern, ground_truth_text)
        if match:
            return float(match.group(1).replace(',', ''))

        # Fallback: try to extract any number
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', ground_truth_text)
        if numbers:
            return float(numbers[-1].replace(',', ''))

        return None

    def evaluate_reasoning(self, response: str) -> float:
        """
        Evaluate quality of reasoning steps.

        Checks for:
        - Presence of step-by-step reasoning
        - Use of mathematical operations
        - Logical flow indicators

        Args:
            response: Model generated response

        Returns:
            Reasoning score between 0 and 1
        """
        score = 0.0

        # Check for step indicators
        step_indicators = [
            r'step \d+', r'first,?', r'then,?', r'next,?',
            r'finally,?', r'therefore,?', r'so,?'
        ]
        for pattern in step_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.2
                break

        # Check for mathematical operations
        math_operations = [r'\+', r'-', r'\*', r'/', r'=', r'×', r'÷']
        for op in math_operations:
            if re.search(op, response):
                score += 0.2
                break

        # Check for explanation words
        explanation_words = ['because', 'since', 'given', 'means', 'equals']
        for word in explanation_words:
            if word in response.lower():
                score += 0.2
                break

        # Check response length (longer usually means more reasoning)
        if len(response) > 100:
            score += 0.2

        # Check for calculation intermediate results
        if re.search(r'\d+\s*[+\-*/×÷]\s*\d+\s*=\s*\d+', response):
            score += 0.2

        return min(score, 1.0)

    def calculate_reward(
        self,
        response: str,
        ground_truth: str,
        **kwargs
    ) -> RewardResult:
        """
        Calculate reward for a model response.

        Args:
            response: Model generated response
            ground_truth: Ground truth answer
            **kwargs: Additional arguments

        Returns:
            RewardResult with score and details
        """
        predicted = self.extract_answer(response)
        expected = self.extract_ground_truth(ground_truth)

        details = {
            "predicted": predicted,
            "expected": expected,
            "answer_correct": False,
            "reasoning_score": 0.0,
        }

        # If we can't extract answers, give low score
        if predicted is None or expected is None:
            logger.debug(
                f"Failed to extract answer. Predicted: {predicted}, Expected: {expected}"
            )
            return RewardResult(score=self.wrong_answer_score, details=details)

        # Check if answer is correct (with small tolerance for floating point)
        answer_correct = abs(predicted - expected) < 1e-6
        details["answer_correct"] = answer_correct

        if answer_correct:
            base_score = self.correct_answer_score
        else:
            base_score = self.wrong_answer_score

        # Add partial credit for reasoning if enabled
        if self.partial_credit:
            reasoning_score = self.evaluate_reasoning(response)
            details["reasoning_score"] = reasoning_score

            # Blend answer and reasoning scores
            final_score = (
                base_score * (1 - self.reasoning_weight) +
                reasoning_score * self.reasoning_weight
            )
        else:
            final_score = base_score

        return RewardResult(score=final_score, details=details)

    def batch_calculate_rewards(
        self,
        responses: List[str],
        ground_truths: List[str],
    ) -> List[RewardResult]:
        """
        Calculate rewards for a batch of responses.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers

        Returns:
            List of RewardResult objects
        """
        assert len(responses) == len(ground_truths), \
            "Number of responses must match ground truths"

        return [
            self.calculate_reward(resp, gt)
            for resp, gt in zip(responses, ground_truths)
        ]


class NewsRewardFunction:
    """
    Reward function for news data evaluation.

    Evaluates news content based on:
    1. Coherence and fluency
    2. Factual consistency
    3. Relevance to input
    4. Style and tone appropriateness

    Note: This is a template to be extended with actual news quality metrics.
    """

    def __init__(
        self,
        coherence_weight: float = 0.3,
        factuality_weight: float = 0.4,
        relevance_weight: float = 0.2,
        style_weight: float = 0.1,
    ):
        """
        Initialize news reward function.

        Args:
            coherence_weight: Weight for coherence score
            factuality_weight: Weight for factual consistency
            relevance_weight: Weight for relevance to prompt
            style_weight: Weight for style/tone
        """
        self.coherence_weight = coherence_weight
        self.factuality_weight = factuality_weight
        self.relevance_weight = relevance_weight
        self.style_weight = style_weight

        # Normalize weights
        total = (coherence_weight + factuality_weight +
                relevance_weight + style_weight)
        self.coherence_weight /= total
        self.factuality_weight /= total
        self.relevance_weight /= total
        self.style_weight /= total

    def evaluate_coherence(self, text: str) -> float:
        """
        Evaluate text coherence and fluency.

        Simple heuristics (to be replaced with actual models):
        - Sentence structure
        - Paragraph organization
        - Transition words

        Args:
            text: Generated news text

        Returns:
            Coherence score between 0 and 1
        """
        score = 0.0

        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            score += 0.3

        # Check for transition words
        transitions = ['however', 'moreover', 'furthermore', 'additionally',
                      'meanwhile', 'consequently', 'therefore']
        if any(word in text.lower() for word in transitions):
            score += 0.3

        # Check for paragraph structure (multiple newlines)
        if '\n\n' in text or text.count('\n') >= 2:
            score += 0.2

        # Check for appropriate length
        if 100 <= len(text) <= 1000:
            score += 0.2

        return min(score, 1.0)

    def evaluate_factuality(self, text: str, reference: str) -> float:
        """
        Evaluate factual consistency with reference.

        Simple heuristics (to be replaced with actual fact-checking models):
        - Named entity overlap
        - Key fact preservation

        Args:
            text: Generated text
            reference: Reference text or facts

        Returns:
            Factuality score between 0 and 1
        """
        # Placeholder: Simple word overlap
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 1.0

        overlap = len(text_words & ref_words) / len(ref_words)
        return min(overlap, 1.0)

    def evaluate_relevance(self, text: str, prompt: str) -> float:
        """
        Evaluate relevance to input prompt.

        Args:
            text: Generated text
            prompt: Original prompt

        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword overlap
        prompt_words = set(prompt.lower().split())
        text_words = set(text.lower().split())

        if not prompt_words:
            return 1.0

        overlap = len(text_words & prompt_words) / len(prompt_words)
        return min(overlap * 1.5, 1.0)  # Scale up slightly

    def evaluate_style(self, text: str) -> float:
        """
        Evaluate writing style and tone.

        Args:
            text: Generated text

        Returns:
            Style score between 0 and 1
        """
        score = 0.0

        # Check for professional news style
        if not re.search(r'[!]{2,}', text):  # No excessive exclamation
            score += 0.3

        # Check for proper capitalization
        if text[0].isupper():
            score += 0.2

        # Check for appropriate vocabulary (presence of news-related words)
        news_words = ['reported', 'announced', 'stated', 'according',
                     'officials', 'sources']
        if any(word in text.lower() for word in news_words):
            score += 0.3

        # Check for formal tone (no slang)
        slang = ['gonna', 'wanna', 'yeah', 'nope']
        if not any(word in text.lower() for word in slang):
            score += 0.2

        return min(score, 1.0)

    def calculate_reward(
        self,
        response: str,
        prompt: str,
        reference: Optional[str] = None,
        **kwargs
    ) -> RewardResult:
        """
        Calculate reward for news generation.

        Args:
            response: Generated news text
            prompt: Original prompt
            reference: Reference text for factuality check
            **kwargs: Additional arguments

        Returns:
            RewardResult with score and details
        """
        coherence = self.evaluate_coherence(response)
        relevance = self.evaluate_relevance(response, prompt)
        style = self.evaluate_style(response)

        # Factuality requires reference
        if reference:
            factuality = self.evaluate_factuality(response, reference)
        else:
            factuality = 0.5  # Neutral score if no reference

        # Calculate weighted score
        final_score = (
            coherence * self.coherence_weight +
            factuality * self.factuality_weight +
            relevance * self.relevance_weight +
            style * self.style_weight
        )

        details = {
            "coherence": coherence,
            "factuality": factuality,
            "relevance": relevance,
            "style": style,
        }

        return RewardResult(score=final_score, details=details)


def get_reward_function(task_type: str = "gsm8k", **kwargs):
    """
    Factory function to get appropriate reward function.

    Args:
        task_type: Type of task ('gsm8k' or 'news')
        **kwargs: Arguments to pass to reward function constructor

    Returns:
        Reward function instance
    """
    if task_type.lower() == "gsm8k":
        return GSM8KRewardFunction(**kwargs)
    elif task_type.lower() == "news":
        return NewsRewardFunction(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
