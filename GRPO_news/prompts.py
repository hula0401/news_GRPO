"""
System prompts for different tasks and reward formats.

This module contains all prompts used during training, separated from the 
training configuration for easy customization.
"""

# News Classification prompt
NEWS_CLASSIFICATION_PROMPT = """Determine whether a news article is EVENT-BASED.

EVENT-BASED news refers to articles reporting specific, concrete events including earnings announcements, mergers and acquisitions, product launches and general economic/political events.
NOT EVENT-BASED news includes general market commentary and analysis, opinion pieces and predictions, sector trends without specific events, historical retrospectives.
Format your response as:
<think>
[put ALL your reasoning here. Consider: What is the main subject? Is there a specific event?]
</think>
<answer>
[true/false - true if event-based, false if not]
</answer>

Example:
1. News: "Apple announces record Q4 earnings, beating analyst expectations with $90B revenue"
Answer:
<think>
This article reports a specific event - Apple's Q4 earnings announcement.
</think>
<answer>
true
</answer>

2. News: "10 stocks to buy now including Tesla and Apple"
Answer:
<think>
This article is a generic list of stocks to buy, not a specific event.
</think>
<answer>
false
</answer>
"""

def get_news_classification_prompt() -> str:
    """
    Get the news classification system prompt.

    Returns:
        Formatted prompt string for news classification task
    """
    return NEWS_CLASSIFICATION_PROMPT

