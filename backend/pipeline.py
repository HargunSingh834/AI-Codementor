"""
Pipeline — full orchestration of one student message through the system.

Flow:
1. Problem identification (problem detector + optional RAG)
2. Intent classification
3. Code analysis (only for code_help)
4. Prompt building
5. Response generation (streamed or canned)
"""

import os
import re
from typing import AsyncGenerator

from groq import Groq

from backend.agents.problem_detector import detect_problem
from backend.agents.query_rewriter import rewrite_query
from backend.agents.intent_classifier import classify_intent
from backend.agents.code_agent import analyze_code
from backend.rag.search import search_problem, CONFIDENCE_THRESHOLD
from backend.session.store import (
    get_session, set_problem, get_problem, has_problem,
    add_to_history, get_history,
)
from backend.prompts.prompt_builder import build_prompt
from backend.utils.canned_responses import CANNED_RESPONSES, FALLBACK_RESPONSES

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _extract_code(message: str) -> str | None:
    """Try to extract code blocks from a student message."""
    # Look for fenced code blocks
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", message, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)

    # Look for inline code-like patterns (indented blocks, function defs)
    lines = message.split("\n")
    code_lines = [
        l for l in lines
        if l.strip().startswith(("def ", "class ", "for ", "while ", "if ", "return ", "import "))
        or (l.startswith("    ") and l.strip())
    ]
    if len(code_lines) >= 2:
        return "\n".join(code_lines)

    return None


async def process_message(
    conversation_id: str, message: str
) -> AsyncGenerator[str, None]:
    """
    Process a single student message through the complete pipeline.
    Yields response tokens as they are generated (for streaming).
    """
    session = get_session(conversation_id)
    problem = get_problem(conversation_id)

    # =============================================
    # STEP 1: Problem Identification
    # =============================================
    if not has_problem(conversation_id):
        detection = await detect_problem(groq_client, message)

        if detection["type"] == "problem_statement":
            # Student pasted a full problem statement — store parsed version
            parsed = detection.get("parsed_problem", {})
            if parsed:
                problem_data = {
                    "id": "pasted",
                    "title": parsed.get("title", "User-Provided Problem"),
                    "description": parsed.get("description", message),
                    "examples": parsed.get("examples", ""),
                    "constraints": parsed.get("constraints", ""),
                    "difficulty": parsed.get("difficulty", "Unknown"),
                    "tags": [],
                    "function_signature": "",
                    "solutions": "",
                }
            else:
                # Fallback: store raw message as the problem
                problem_data = {
                    "id": "pasted",
                    "title": "User-Provided Problem",
                    "description": message,
                    "examples": "",
                    "constraints": "",
                    "difficulty": "Unknown",
                    "tags": [],
                    "function_signature": "",
                    "solutions": "",
                }
            set_problem(conversation_id, problem_data)
            problem = problem_data

            # Acknowledge problem received
            ack = f"Got it! I can see you're working on **{problem_data['title']}**. What part are you stuck on or would you like to discuss?"
            add_to_history(conversation_id, "user", message)
            add_to_history(conversation_id, "assistant", ack)
            yield ack
            return

        elif detection["type"] == "problem_reference":
            # Student referenced a problem — try RAG search
            query = detection.get("extracted_query", "")
            if not query:
                query = await rewrite_query(groq_client, message)

            problem_data, score = search_problem(query)

            if problem_data and score >= CONFIDENCE_THRESHOLD:
                # High confidence match
                set_problem(conversation_id, problem_data)
                problem = problem_data
                title = problem_data.get("title", "this problem")
                ack = f"Great, you're working on **{title}**! How can I help you with it? Are you trying to understand the problem, figure out an approach, or debug some code?"
                add_to_history(conversation_id, "user", message)
                add_to_history(conversation_id, "assistant", ack)
                yield ack
                return

            elif problem_data and score > 0:
                # Low confidence match
                title = problem_data.get("title", "unknown")
                resp = FALLBACK_RESPONSES["low_confidence_match"].format(title=title)
                add_to_history(conversation_id, "user", message)
                add_to_history(conversation_id, "assistant", resp)
                yield resp
                return

            else:
                # Not found
                resp = FALLBACK_RESPONSES["not_found"]
                add_to_history(conversation_id, "user", message)
                add_to_history(conversation_id, "assistant", resp)
                yield resp
                return

        else:
            # No problem referenced
            resp = FALLBACK_RESPONSES["no_problem_referenced"]
            add_to_history(conversation_id, "user", message)
            add_to_history(conversation_id, "assistant", resp)
            yield resp
            return

    # =============================================
    # STEP 2: Intent Classification
    # =============================================
    problem_title = problem.get("title", "Unknown") if problem else "Unknown"
    problem_desc = problem.get("description", "")[:200] if problem else ""

    intent = await classify_intent(
        groq_client, message, problem_title, problem_desc
    )

    category = intent.get("category", "off_topic")
    sub_intent = intent.get("sub_intent", "unclear")

    # =============================================
    # STEP 2b: Off-topic → canned response (no LLM)
    # =============================================
    if category == "off_topic":
        resp = CANNED_RESPONSES.get(sub_intent, CANNED_RESPONSES["unclear"])
        add_to_history(conversation_id, "user", message)
        add_to_history(conversation_id, "assistant", resp)
        yield resp
        return

    # =============================================
    # STEP 3: Code Agent (only for code_help)
    # =============================================
    code_analysis = None
    if category == "code_help":
        student_code = _extract_code(message) or message
        solution_approach = problem.get("solutions", "") if problem else ""
        problem_description = problem.get("description", "") if problem else ""

        code_analysis = await analyze_code(
            groq_client,
            student_code=student_code,
            problem_description=problem_description,
            solution_approach=solution_approach,
            sub_intent=sub_intent,
        )

    # =============================================
    # STEP 4: Build prompt + Generate response
    # =============================================
    messages = build_prompt(
        session=session,
        intent=intent,
        code_analysis=code_analysis,
        student_message=message,
    )

    # Stream the response
    full_response = []
    stream = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=400,
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_response.append(token)
            yield token

    # Save to history
    response_text = "".join(full_response)
    add_to_history(conversation_id, "user", message)
    add_to_history(conversation_id, "assistant", response_text)
