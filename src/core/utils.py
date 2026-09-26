"""
Lab 11 — Helper Utilities
"""
import asyncio
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Includes automatic retry with exponential backoff on transient errors (503 / 429).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            session = None
            if session_id is not None:
                try:
                    session = await runner.session_service.get_session(
                        app_name=app_name, user_id=user_id, session_id=session_id
                    )
                except (ValueError, KeyError):
                    pass

            if session is None:
                try:
                    session = await runner.session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )
                except Exception:
                    session = await runner.session_service.create_session(
                        app_name=app_name, user_id=user_id
                    )

            content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )

            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text

            return final_response, session

        except Exception as e:
            err_str = str(e).lower()
            if (
                "503" in err_str
                or "unavailable" in err_str
                or "high demand" in err_str
                or "resource_exhausted" in err_str
                or "429" in err_str
            ) and attempt < max_retries:
                wait_time = (attempt + 1) * 3  # 3s, 6s, 9s
                print(
                    f"[Retry {attempt + 1}/{max_retries}] Gemini server busy. Waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                continue
            if (
                "503" in err_str
                or "unavailable" in err_str
                or "high demand" in err_str
                or "dynamic node" in err_str
                or "node execution failed" in err_str
            ):
                print(f"Gemini API temporary overload. Returning fallback response: {e}")
                return "I cannot provide that information at this moment due to service unavailability.", session
            raise
