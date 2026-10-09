"""
Ingest LeetCode dataset into ChromaDB.

Usage:
    cd code_tutor
    source venv/bin/activate
    python -m backend.rag.ingest --dataset data/leetcode_dataset.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rag.chromadb_setup import get_collection


def extract_function_signature(code_snippets) -> str:
    """Extract Python function signature from code_snippets dict."""
    if not code_snippets or not isinstance(code_snippets, dict):
        return ""

    # Prefer python3 over python
    code = code_snippets.get("python3", code_snippets.get("python", ""))
    if not code:
        return ""

    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else ""


def format_examples(examples) -> str:
    """Format examples list into a readable string."""
    if not examples:
        return ""
    if isinstance(examples, str):
        return examples
    if isinstance(examples, list):
        parts = []
        for ex in examples:
            if isinstance(ex, dict):
                parts.append(ex.get("example_text", str(ex)))
            else:
                parts.append(str(ex))
        return "\n\n".join(parts)
    return str(examples)


def format_constraints(constraints) -> str:
    """Format constraints list into a readable string."""
    if not constraints:
        return ""
    if isinstance(constraints, str):
        return constraints
    if isinstance(constraints, list):
        return "\n".join(f"• {c}" for c in constraints)
    return str(constraints)


def ingest_dataset(dataset_path: str):
    """Load the LeetCode dataset into ChromaDB."""
    print(f"Loading dataset from {dataset_path}...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dataset is {"questions": [...]}
    problems = data.get("questions", data) if isinstance(data, dict) else data
    if not isinstance(problems, list):
        problems = list(problems.values()) if isinstance(problems, dict) else [problems]

    print(f"Found {len(problems)} problems")

    collection = get_collection()

    # Process in batches
    batch_size = 100
    ids = []
    documents = []
    metadatas = []

    for i, problem in enumerate(problems):
        if not isinstance(problem, dict):
            continue

        # Extract fields from the actual dataset schema
        pid = str(problem.get("frontend_id", problem.get("problem_id", i)))
        title = problem.get("title", "")
        difficulty = problem.get("difficulty", "Unknown")
        description = problem.get("description", "")
        topics = problem.get("topics", [])
        examples = problem.get("examples", [])
        constraints = problem.get("constraints", [])
        hints = problem.get("hints", [])
        code_snippets = problem.get("code_snippets", {})
        solution = problem.get("solution", "")
        problem_slug = problem.get("problem_slug", "")

        # Extract function signature from Python code
        func_sig = extract_function_signature(code_snippets)

        # Build embed text for vector search
        desc_truncated = description[:500] if description else ""
        embed_text = f"{pid} {title} {func_sig} {desc_truncated}"

        # Format examples and constraints as strings
        examples_str = format_examples(examples)
        constraints_str = format_constraints(constraints)

        # Build metadata (stored alongside the vector)
        metadata = {
            "id": pid,
            "title": title,
            "difficulty": difficulty,
            "problem_slug": problem_slug,
            "topics": json.dumps(topics) if isinstance(topics, list) else str(topics),
            "description": description[:5000] if description else "",
            "examples": examples_str[:3000] if examples_str else "",
            "constraints": constraints_str[:1000] if constraints_str else "",
            "hints": json.dumps(hints) if isinstance(hints, list) else str(hints),
            "function_signature": func_sig,
            "solutions": str(solution)[:3000] if solution else "",
        }

        ids.append(f"problem_{pid}")
        documents.append(embed_text)
        metadatas.append(metadata)

        # Batch upsert
        if len(ids) >= batch_size:
            print(f"  Upserting problems {i - batch_size + 2}-{i + 1}...")
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            ids, documents, metadatas = [], [], []

    # Final batch
    if ids:
        print(f"  Upserting final batch ({len(ids)} problems)...")
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    total = collection.count()
    print(f"✅ Done! {total} problems indexed in ChromaDB.")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest LeetCode dataset into ChromaDB")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    args = parser.parse_args()
    ingest_dataset(args.dataset)


if __name__ == "__main__":
    main()
