from GRPO.gsm8k_reward import compute_score

def main() -> None:
    """Simple smoke test for the custom GSM8K reward."""
    solution = "Natalia sold ... #### 72"
    ground_truth = "72"
    reward = compute_score(
        data_source="gsm8k",
        solution_str=solution,
        ground_truth=ground_truth,
        extra_info=None,
    )
    print(f"Reward: {reward}")

if __name__ == "__main__":
    main()
