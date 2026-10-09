"""
Chainlit frontend for CodeMentor AI.
Connects to the FastAPI backend /chat endpoint and streams responses.
"""

import os
import uuid
import httpx
import chainlit as cl

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new conversation session."""
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)

    await cl.Message(
        content=(
            "👋 **Welcome to CodeMentor AI!**\n\n"
            "I'm your Socratic coding tutor — I help you solve LeetCode problems "
            "by guiding your thinking, never by giving you the answer.\n\n"
            "**How to get started:**\n"
            "- Tell me which LeetCode problem you're working on (e.g., \"I'm working on Two Sum\")\n"
            "- Or paste the full problem statement\n"
            "- Or mention the problem number (e.g., \"problem 121\")\n\n"
            "Once I know the problem, ask me anything — I'll help you think through it! 🧠"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process a student message through the backend pipeline."""
    conversation_id = cl.user_session.get("conversation_id")

    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/chat",
                json={
                    "conversation_id": conversation_id,
                    "message": message.content,
                },
            ) as response:
                if response.status_code != 200:
                    msg.content = "Sorry, something went wrong. Please try again."
                    await msg.update()
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        token = line[6:]
                        if token == "[DONE]":
                            break
                        msg.content += token
                        await msg.update()

    except httpx.ConnectError:
        msg.content = (
            "⚠️ Cannot connect to the backend server. "
            "Make sure the backend is running on " + BACKEND_URL
        )
        await msg.update()
    except Exception as e:
        msg.content = f"⚠️ Error: {str(e)}"
        await msg.update()


@cl.on_chat_end
async def on_chat_end():
    """Clean up session on chat end."""
    conversation_id = cl.user_session.get("conversation_id")
    if conversation_id:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{BACKEND_URL}/session/reset",
                    json={"conversation_id": conversation_id},
                )
        except Exception:
            pass
