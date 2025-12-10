import re
from verl.utils.reward_score import RewardFunction

class GSM8KReward(RewardFunction):
    """
    Exact-match reward for GSM8K.
    Extract the final numeric answer ("#### <number>")
    from model output and compare with ground truth.
    """

    # Regex to match: #### 72
    answer_pattern = re.compile(r"####\s*([-+]?\d+\.?\d*)")

    def extract_answer(self, text: str):
        """Extract the final number after ####."""
        if text is None:
            return None
        m = self.answer_pattern.search(text)
        if m:
            return m.group(1).strip()
        return None

    def compute(self, data):
        """
        Args:
            data["model_output"]: model-generated text (string)
            data["answer"]: ground truth answer string
        """
        model_output = data.get("model_output", "")
        gt_answer = data.get("answer", "").strip()

        pred = self.extract_answer(model_output)

        # If model failed to output a valid answer → reward = 0
        if pred is None or gt_answer is None:
            return 0.0

        # Compare exact match
        return float(pred == gt_answer)
