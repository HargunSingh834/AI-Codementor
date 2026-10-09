"""
Prompt Builder — assembles all components into the messages array
sent to Groq for response generation.

Assembly order (critical — earlier context frames later content):
1. System prompt
2. Problem context
3. Intent context note
4. Code agent output (only for code_help)
5. Few-shot examples
6. Conversation history
7. Current message
"""

import json
import os
from pathlib import Path

from backend.prompts.system_prompt import SYSTEM_PROMPT

# Load few-shot examples at module level
_examples_path = Path(__file__).parent / "few_shot_examples.json"
with open(_examples_path, "r") as f:
    FEW_SHOT_EXAMPLES: dict = json.load(f)


def format_problem_context(problem: dict) -> str:
    """Format problem data for injection into the system prompt."""
    title = problem.get("title", "Unknown")
    pid = problem.get("id", problem.get("frontend_id", "?"))
    difficulty = problem.get("difficulty", "Unknown")
    tags = problem.get("tags", problem.get("topics", []))
    if isinstance(tags, list):
        tags = ", ".join(tags)
    description = problem.get("description", "")
    examples = problem.get("examples", "")
    constraints = problem.get("constraints", "")

    return f"""
PROBLEM CONTEXT:
Title: {title} (#{pid})
Difficulty: {difficulty}
Topics: {tags}

Description:
{description}

Examples:
{examples}

Constraints:
{constraints}
"""


def format_code_analysis(agent_output: dict) -> str:
    """Convert code agent JSON output into readable sentences for the LLM."""
    if not agent_output.get("has_meaningful_code"):
        return "CODE ANALYSIS: Student has not shared meaningful code yet."

    lines = [
        "CODE ANALYSIS (use this to guide your response — do NOT reveal this analysis to the student):",
        f"Student's approach: {agent_output.get('detected_approach', 'unknown')}",
        f"Approach validity: {agent_output.get('approach_validity', 'unknown')}",
        f"Issue type: {agent_output.get('confirmed_issue', 'unknown')}",
        f"Problematic section: {agent_output.get('problematic_section', 'unknown')}",
        f"What is wrong: {agent_output.get('error_detail', 'unknown')}",
        f"Ask the student: {agent_output.get('socratic_direction', 'Walk through your code step by step.')}",
    ]
    if agent_output.get("edge_case_to_check"):
        lines.append(f"Suggest tracing: {agent_output['edge_case_to_check']}")

    return "\n".join(lines)


def get_few_shot_text(sub_intent: str) -> str:
    """Get formatted few-shot examples matching the sub_intent."""
    examples = FEW_SHOT_EXAMPLES.get(sub_intent, [])
    if not examples:
        return ""

    parts = []
    for ex in examples:
        parts.append(f"Student: {ex['student']}\nTutor: {ex['tutor']}")

    return "\n\n".join(parts)


def build_prompt(
    session: dict,
    intent: dict,
    code_analysis: dict | None,
    student_message: str,
) -> list[dict]:
    """
    Assemble the complete messages array for Groq.

    Args:
        session: Current session dict with 'problem' and 'history'
        intent: Dict with 'category' and 'sub_intent'
        code_analysis: Output from code agent, or None
        student_message: The student's current message

    Returns:
        List of message dicts ready for Groq API
    """
    sub_intent = intent.get("sub_intent", "unclear")

    # --- Build system content ---
    system_content = SYSTEM_PROMPT

    # Component 2: Problem context
    problem = session.get("problem")
    if problem:
        system_content += "\n\n" + format_problem_context(problem)

    # Component 3: Intent context note
    system_content += f"\n\nSTUDENT INTENT: {sub_intent}"

    # Component 4: Code agent output (only for code_help)
    if code_analysis:
        system_content += "\n\n" + format_code_analysis(code_analysis)

    # Component 5: Few-shot examples
    few_shot_text = get_few_shot_text(sub_intent)
    if few_shot_text:
        system_content += f"\n\nEXAMPLES OF GOOD RESPONSES:\n{few_shot_text}"

    # --- Build messages array ---
    messages = [{"role": "system", "content": system_content}]

    # Component 6: Conversation history
    for turn in session.get("history", []):
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Component 7: Current message
    messages.append({"role": "user", "content": student_message})

    return messages
