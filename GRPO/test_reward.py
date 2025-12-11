"""
Test script for reward functions

Run this to verify reward functions work correctly before training.
"""

import logging
from reward_function import GSM8KRewardFunction, NewsRewardFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gsm8k_reward():
    """Test GSM8K reward function with various cases"""
    print("\n" + "=" * 70)
    print("Testing GSM8K Reward Function")
    print("=" * 70)

    reward_fn = GSM8KRewardFunction(
        correct_answer_score=1.0,
        wrong_answer_score=0.0,
        partial_credit=True,
        reasoning_weight=0.2
    )

    test_cases = [
        {
            "name": "Correct answer with good reasoning",
            "response": (
                "Let me solve this step by step.\n"
                "First, eggs laid per day: 16\n"
                "Eggs eaten: 3\n"
                "Eggs used for muffins: 4\n"
                "Remaining: 16 - 3 - 4 = 9\n"
                "Price per egg: $2\n"
                "Total: 9 × 2 = 18\n"
                "#### 18"
            ),
            "ground_truth": "Janet sells 16 - 3 - 4 = 9 eggs\n#### 18"
        },
        {
            "name": "Correct answer with minimal reasoning",
            "response": "The answer is 18\n#### 18",
            "ground_truth": "#### 18"
        },
        {
            "name": "Wrong answer with good reasoning",
            "response": (
                "Step 1: 16 eggs per day\n"
                "Step 2: Used 3 + 4 = 7 eggs\n"
                "Step 3: Remaining 16 - 7 = 9\n"
                "Step 4: Income = 9 × 3 = 27\n"
                "#### 27"
            ),
            "ground_truth": "#### 18"
        },
        {
            "name": "Wrong answer with no reasoning",
            "response": "#### 42",
            "ground_truth": "#### 18"
        },
        {
            "name": "No answer extracted",
            "response": "I don't know how to solve this problem.",
            "ground_truth": "#### 18"
        },
        {
            "name": "Answer with commas",
            "response": "The total is 1,234 dollars.\n#### 1,234",
            "ground_truth": "#### 1234"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 70)

        result = reward_fn.calculate_reward(
            response=case["response"],
            ground_truth=case["ground_truth"]
        )

        print(f"Response preview: {case['response'][:100]}...")
        print(f"Reward Score: {result.score:.3f}")
        print(f"Details: {result.details}")


def test_news_reward():
    """Test News reward function with various cases"""
    print("\n" + "=" * 70)
    print("Testing News Reward Function")
    print("=" * 70)

    reward_fn = NewsRewardFunction(
        coherence_weight=0.3,
        factuality_weight=0.4,
        relevance_weight=0.2,
        style_weight=0.1
    )

    test_cases = [
        {
            "name": "High quality news article",
            "response": (
                "Tech companies announced significant breakthroughs in artificial "
                "intelligence development. According to industry sources, these "
                "advances represent major progress in machine learning.\n\n"
                "Moreover, officials stated that the innovations will transform "
                "computing capabilities. Industry experts reported positive "
                "reactions to the announcements."
            ),
            "prompt": "Write about recent AI developments in tech companies",
            "reference": "AI breakthroughs announced by tech companies"
        },
        {
            "name": "Low quality - no structure",
            "response": "yeah tech stuff is cool gonna be big",
            "prompt": "Write about recent AI developments",
            "reference": "AI developments"
        },
        {
            "name": "Medium quality - some issues",
            "response": (
                "There were some announcements!!! Companies did things. "
                "AI is here and it's amazing!!! Everything will change!!!"
            ),
            "prompt": "Write about AI announcements",
            "reference": "AI announcements by companies"
        },
        {
            "name": "Good coherence but off-topic",
            "response": (
                "The weather forecast indicates rain this weekend. "
                "Meteorologists reported changing conditions. "
                "Therefore, residents should prepare accordingly."
            ),
            "prompt": "Write about recent AI developments",
            "reference": "AI developments"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 70)

        result = reward_fn.calculate_reward(
            response=case["response"],
            prompt=case["prompt"],
            reference=case.get("reference")
        )

        print(f"Response preview: {case['response'][:100]}...")
        print(f"Reward Score: {result.score:.3f}")
        print(f"Details:")
        for key, value in result.details.items():
            print(f"  {key}: {value:.3f}")


def test_batch_processing():
    """Test batch processing of rewards"""
    print("\n" + "=" * 70)
    print("Testing Batch Processing")
    print("=" * 70)

    reward_fn = GSM8KRewardFunction()

    responses = [
        "Step by step solution\n#### 18",
        "Quick answer\n#### 18",
        "Wrong calculation\n#### 42",
    ]
    ground_truths = ["#### 18"] * 3

    results = reward_fn.batch_calculate_rewards(responses, ground_truths)

    print(f"\nProcessed {len(results)} responses")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.3f}, Correct: {result.details['answer_correct']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REWARD FUNCTION TEST SUITE")
    print("=" * 70)

    test_gsm8k_reward()
    test_news_reward()
    test_batch_processing()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
