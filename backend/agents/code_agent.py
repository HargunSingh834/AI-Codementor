"""
Code Agent (Agent 3) — analyzes student code against the problem.
Fires only when intent category is code_help.
Outputs structured JSON with socratic_direction — never the solution.
"""

import json
from groq import Groq

CODE_AGENT_PROMPT = """You are a code analysis engine for a coding tutoring system.
Analyze the student code against the problem. Do NOT suggest fixes.
Do NOT reveal the correct approach or solution. Return ONLY valid JSON.

Problem: {problem_description}
Correct approach (for your reference only — never reveal to student): {solution_approach}
Note: This is ONE possible correct approach. The student may be using a different but equally valid approach. Evaluate their approach on its own merits.

Student code:
```
{student_code}
```

Suspected issue type: {sub_intent}

Return this exact JSON:
{{
  "has_meaningful_code": true | false,
  "detected_approach": "brief description of what student is attempting",
  "approach_validity": "correct" | "incorrect" | "suboptimal",
  "confirmed_issue": "syntax_error" | "runtime_error" | "incorrect_behavior" | "edge_case_failure" | "partial_correctness" | "time_complexity" | "space_complexity",
  "problematic_section": "specific line or section, no solution revealed",
  "error_detail": "what is wrong in one sentence — no fix given",
  "socratic_direction": "what question to ask the student to find it themselves",
  "edge_case_to_check": "specific small input to trace manually, or null"
}}"""


async def analyze_code(
    client: Groq,
    student_code: str,
    problem_description: str,
    solution_approach: str,
    sub_intent: str,
) -> dict:
    """
    Analyze student code and return structured diagnosis.
    Never contains the solution or fix — only socratic_direction.
    """
    prompt = CODE_AGENT_PROMPT.format(
        problem_description=problem_description,
        solution_approach=solution_approach or "No editorial solution available. Analyze the code independently.",
        student_code=student_code,
        sub_intent=sub_intent,
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
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
        result = {
            "has_meaningful_code": True,
            "detected_approach": "Unable to parse — manual review needed",
            "approach_validity": "unknown",
            "confirmed_issue": sub_intent,
            "problematic_section": "unknown",
            "error_detail": "Code agent could not produce structured analysis",
            "socratic_direction": "Can you walk me through your code step by step? What does each part do?",
            "edge_case_to_check": None,
        }

    return result
