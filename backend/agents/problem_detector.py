"""
Problem Detector Agent — replaces rule-based Steps 2 & 3 from the spec.
Uses an LLM call to determine if a message contains a pasted problem
statement, references a known problem, or neither.
"""

import json
from groq import Groq

PROBLEM_DETECTOR_PROMPT = """You are analyzing a student message in a coding tutoring system.
Determine what kind of problem information this message contains.

Student message:
{message}

Classify into ONE of these types:
1. "problem_statement" — The student has pasted a full problem statement (contains description, examples, constraints, input/output format). This is typically 200+ characters with structured problem details.
2. "problem_reference" — The student is referencing a known LeetCode problem by number, title, function name, or brief description (e.g. "I'm working on Two Sum", "problem 42", "the one about islands in a grid", "def twoSum").
3. "none" — The student is NOT referencing any specific problem. They might be asking a general question, saying hello, or their message doesn't mention any problem.

If type is "problem_reference", also extract a clean 2-5 word search query that describes the problem (the title or core concept).

If type is "problem_statement", extract these fields from the pasted text:
- title: problem title if mentioned, otherwise generate a short descriptive title
- description: the main problem description
- examples: the examples section
- constraints: the constraints section
- difficulty: if mentioned, otherwise "Unknown"

Return ONLY valid JSON in this exact format:
{{"type": "problem_statement" | "problem_reference" | "none", "extracted_query": "search query or null", "parsed_problem": {{...}} or null}}

Examples:
- "I'm working on problem 1" → {{"type": "problem_reference", "extracted_query": "two sum array target", "parsed_problem": null}}
- "what is DP?" → {{"type": "none", "extracted_query": null, "parsed_problem": null}}
- "Given an array of integers nums and an integer target, return indices... Example 1: Input: nums = [2,7,11,15]..." → {{"type": "problem_statement", "extracted_query": null, "parsed_problem": {{"title": "Two Sum", "description": "Given an array...", "examples": "Example 1:...", "constraints": "...", "difficulty": "Easy"}}}}"""


async def detect_problem(client: Groq, message: str) -> dict:
    """
    Detect whether the student message contains/references a problem.
    Returns dict with keys: type, extracted_query, parsed_problem
    """
    prompt = PROBLEM_DETECTOR_PROMPT.format(message=message)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Try to extract JSON from the response
    try:
        # Handle potential markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: assume no problem detected
        result = {"type": "none", "extracted_query": None, "parsed_problem": None}

    # Ensure all required keys exist
    result.setdefault("type", "none")
    result.setdefault("extracted_query", None)
    result.setdefault("parsed_problem", None)

    return result
