from verl.utils.reward_score.gsm8k_reward import GSM8KReward

reward_fn = GSM8KReward()

sample = {
    "model_output": "I think the answer is 72.\n#### 72",
    "answer": "72"
}

print(reward_fn.compute(sample))  # Should print 1.0
