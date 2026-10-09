"""
Query Rewriter Agent (Agent 1) — extracts a clean search query
from the student message for ChromaDB vector search.
"""

from groq import Groq

QUERY_REWRITER_PROMPT = """Extract the LeetCode problem being referenced from this student message.
Return ONLY a short search query (2-5 words) — the problem title or \
a brief description of what the problem asks. Nothing else.

Examples:
Message: 'I am working on problem 1' → 'two sum array target'
Message: 'def twoSum(self, nums, target):' → 'two sum array target'
Message: 'the one where you find islands in a grid' → 'number of islands grid'
Message: 'Two Sum' → 'two sum array target'
Message: 'problem 121' → 'best time buy sell stock'
Message: 'I need help with valid parentheses' → 'valid parentheses stack'
Message: 'the climbing stairs problem' → 'climbing stairs ways'
Message: 'binary search in a rotated array' → 'search rotated sorted array'
Message: 'merge two sorted linked lists' → 'merge two sorted lists'
Message: 'how do I reverse a linked list' → 'reverse linked list'
Message: 'maximum subarray problem' → 'maximum subarray kadane'
Message: 'coin change minimum coins' → 'coin change dynamic programming'

Student message: {message}
Search query:"""


async def rewrite_query(client: Groq, message: str) -> str:
    """
    Rewrite a student message into a clean 2-5 word search query
    optimized for ChromaDB vector search.
    """
    prompt = QUERY_REWRITER_PROMPT.format(message=message)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.1,
    )

    query = response.choices[0].message.content.strip()
    # Remove any quotes or extra formatting
    query = query.strip("'\"")
    return query
