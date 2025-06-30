import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables from .env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in .env.")

@cl.on_chat_start
async def start():
    # Initialize Gemini-compatible OpenAI client
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Model config
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    # Save to user session
    config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

    # Welcome message
    await cl.Message(
        content="""
# 🤖 Panaversity AI Assistant  
**Created by Saif Soomro**  
Welcome! How can I help you today?
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Send placeholder
    msg = cl.Message(content="")
    await msg.send()

    # Get chat history
    history = cl.user_session.get("chat_history") or []

    # Add system message (with instructions) if not already present
    if not any(m.get("role") == "system" for m in history):
        history.insert(0, {
            "role": "system",
            "content": """
You are a helpful and intelligent assistant created by **Saif Soomro**.
If a user asks who created you, proudly respond with "I was created by **Saif Soomro**."
Always be polite, informative, and concise.
"""
        })

    # Add user input to history
    history.append({"role": "user", "content": message.content})

    # Load model & client from session
    external_client = cl.user_session.get("config").model_provider
    model = cl.user_session.get("config").model

    try:
        # Call Gemini API with streaming
        stream = await external_client.chat.completions.create(
            messages=history,
            model=model.model,
            stream=True
        )

        assistant_response = ""
        async for part in stream:
            token = part.choices[0].delta.content
            if token:
                assistant_response += token
                await msg.stream_token(token)

        # Save final response
        history.append({"role": "assistant", "content": assistant_response})
        cl.user_session.set("chat_history", history)

        # Final update
        await msg.update()

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"[Error] {e}")
