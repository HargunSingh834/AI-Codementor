"""
Intent Classifier Agent (Agent 2) — classifies every student message
into a category (off_topic, problem_help, code_help) and a sub_intent.

Uses the user's custom 22 sub-intent taxonomy.
"""

import json
from groq import Groq

INTENT_CLASSIFIER_PROMPT = """You are classifying a student message for a coding tutoring system.
The student is working on a LeetCode problem.

Problem context: {problem_title} — {problem_description_short}
Student message: {message}

Classify into one of these categories:
- off_topic — irrelevant, asking for direct answer, or unclassifiable
- problem_help — help understanding, approaching, or conceptual question about the problem (NO code shared)
- code_help — student has shared code in their message

Also return a sub_intent from the appropriate list:

For off_topic:
- irrelevant: question has nothing to do with coding/the problem
- answer_request: directly demanding solution or answer
- unclear: doesn't fit any other category

For problem_help:
- problem_clarification: doesn't understand what the problem is asking
- example_clarification: confused by the given examples in the problem
- constraints_clarification: doesn't understand the problem constraints
- getting_started: understands problem but has no idea how to begin
- pattern_identification: can't identify what algorithmic pattern applies
- approach_validation: has an idea and wants to confirm before coding
- approach_correction: has committed to a wrong approach
- hint_request: explicitly asks for a hint without revealing answer
- approach_unfamiliar: knows the technique needed but has never used it
- language_gap: knows what to do algorithmically but blocked by syntax/API
- stuck_implementation: knows approach, started, but can't proceed
- logic_translation: has the idea in their head but can't convert it to code

For code_help:
- syntax_error: compile or parse error
- runtime_error: code crashes during execution
- incorrect_behavior: code runs but produces wrong results
- edge_case_failure: passes main tests but fails on boundary/edge inputs
- partial_correctness: passes some test cases but not all
- time_complexity: solution is too slow / TLE
- space_complexity: solution uses too much memory

Return ONLY valid JSON:
{{"category": "...", "sub_intent": "..."}}"""


async def classify_intent(
    client: Groq,
    message: str,
    problem_title: str = "Not yet identified",
    problem_description_short: str = "No problem in session yet",
) -> dict:
    """
    Classify a student message into category + sub_intent.
    Returns dict with keys: category, sub_intent
    """
    prompt = INTENT_CLASSIFIER_PROMPT.format(
        message=message,
        problem_title=problem_title,
        problem_description_short=problem_description_short,
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"category": "off_topic", "sub_intent": "unclear"}

    # Validate category
    valid_categories = {"off_topic", "problem_help", "code_help"}
    if result.get("category") not in valid_categories:
        result["category"] = "off_topic"
        result["sub_intent"] = "unclear"

    return result
