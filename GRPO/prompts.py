"""
System prompts for different tasks and reward formats.

This module contains all prompts used during training, separated from the 
training configuration for easy customization.
"""

# GSM8K prompts for different reward function formats

GSM8K_PROMPT_GRPO_FORMAT = """Answer the above math problem step by step.

Format your response as:
<think>
[Show your step-by-step reasoning here]
</think>
<answer>
[Final numeric answer only]
</answer>

Example:
Question: If John has 5 apples and gives 2 to Mary, how many does he have left?
<think>
John starts with 5 apples.
He gives away 2 apples to Mary.
5 - 2 = 3 apples remaining.
</think>
<answer>
3
</answer>"""

# Default prompt (matches original GSM8K format)
GSM8K_PROMPT_DEFAULT = GSM8K_PROMPT_GRPO_FORMAT


def get_gsm8k_prompt(format_type: str = "grpo") -> str:
    """
    Get the appropriate GSM8K prompt based on reward function format.
    
    Args:
        format_type: One of 'grpo', 'partial', or 'default'
        
    Returns:
        Formatted prompt string
    """
    prompts = {
        "grpo": GSM8K_PROMPT_GRPO_FORMAT,
        "partial": GSM8K_PROMPT_PARTIAL_FORMAT,
        "default": GSM8K_PROMPT_DEFAULT,
    }
    
    return prompts.get(format_type, GSM8K_PROMPT_DEFAULT)

