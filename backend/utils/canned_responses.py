"""
Canned responses for off_topic intents.
No LLM call is made — these are returned directly.
"""

CANNED_RESPONSES = {
    "irrelevant": (
        "That's outside what I can help with! I'm built specifically "
        "to help you work through LeetCode problems. "
        "Which problem are you working on?"
    ),
    "answer_request": (
        "I genuinely can't give you the answer — but I'm here to help "
        "you get there. What's the specific part that feels most "
        "impossible right now? Is it understanding the problem, "
        "the approach, or something in your code?"
    ),
    "unclear": (
        "I'm not quite sure what you're asking. Could you tell me "
        "more about what you're stuck on with this problem?"
    ),
}

# Fallback responses for problem identification flow
FALLBACK_RESPONSES = {
    "no_problem_referenced": (
        "Which LeetCode problem are you working on? You can share the "
        "problem number, title, or paste the problem statement directly."
    ),
    "low_confidence_match": (
        "I think you might be asking about {title}. "
        "Is this the right problem? If not, please paste the problem "
        "statement directly."
    ),
    "not_found": (
        "I couldn't find that problem in my database. Could you paste "
        "the full problem statement directly in the chat?"
    ),
}
