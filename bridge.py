#!/usr/bin/env python3
"""
Sophie Voice Bridge - ElevenLabs Custom LLM Server

Exposes an OpenAI-compatible /v1/chat/completions endpoint that:
1. Uses Haiku (via Clawdbot) for fast voice responses
2. Can escalate to Sophie (Opus) for complex questions via ask_sophie tool
3. Maintains Sophie's persona throughout

Usage:
    python bridge.py
    # Then point ElevenLabs Custom LLM to http://localhost:8013/v1/chat/completions
"""

import json
import os
import asyncio
from typing import Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

# Config - use Clawdbot gateway as the LLM backend
CLAWDBOT_GATEWAY_URL = os.getenv("CLAWDBOT_GATEWAY_URL", "http://localhost:18789")
CLAWDBOT_GATEWAY_TOKEN = os.getenv("CLAWDBOT_GATEWAY_TOKEN", "")

if not CLAWDBOT_GATEWAY_TOKEN:
    print("Warning: CLAWDBOT_GATEWAY_TOKEN not set - calls may fail")

app = FastAPI(title="Sophie Voice Bridge")

# Sophie's voice persona - optimized for voice conversations
SOPHIE_VOICE_SYSTEM = """You are Sophie, Aaron's AI partner. You're speaking on a voice call via WhatsApp.

Key traits:
- Sharp, warm, capable
- Concise and conversational (this is voice, not text!)
- Professional when it matters, fun when it doesn't
- You think ahead and connect dots

Voice-specific guidelines:
- Keep responses SHORT and conversational - you're speaking, not writing
- Aim for 1-3 sentences max for simple questions
- Use natural speech patterns, not formal prose
- Don't use markdown, bullet points, or formatting - it's audio
- Don't say "um" or "uh" but do use natural filler like "well" or "so"
- If something requires research or complex tasks, offer to look into it

Current time: {current_time}
"""

# OpenAI-compatible request/response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str = "claude-3-haiku-20240307"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = True
    user_id: Optional[str] = None
    elevenlabs_extra_body: Optional[dict] = None


async def call_clawdbot(messages: list, model: str = "anthropic/claude-3-haiku-20240307", 
                        max_tokens: int = 1024, system: str = None, stream: bool = False):
    """Call Clawdbot's OpenAI-compatible endpoint."""
    
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Build the messages list with system prompt
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(messages)
    
    payload = {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    if stream:
        # Return streaming response
        async def stream_from_clawdbot():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
                        elif line == "data: [DONE]":
                            yield "data: [DONE]\n\n"
        return stream_from_clawdbot()
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, 
                                  detail=f"Clawdbot error: {response.text}")


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint for ElevenLabs."""
    
    # Build system prompt with current time
    system = SOPHIE_VOICE_SYSTEM.format(
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    )
    
    # Extract any system message from the request and prepend
    for msg in request.messages:
        if msg.role == "system":
            system = msg.content + "\n\n" + system
            break
    
    # Convert messages (skip system, already handled)
    messages = [{"role": m.role, "content": m.content} 
                for m in request.messages if m.role != "system"]
    
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    # Use Haiku for fast voice responses
    model = "anthropic/claude-3-haiku-20240307"
    
    try:
        if request.stream:
            stream_gen = await call_clawdbot(
                messages=messages,
                model=model,
                max_tokens=request.max_tokens or 1024,
                system=system,
                stream=True
            )
            return StreamingResponse(stream_gen, media_type="text/event-stream")
        else:
            result = await call_clawdbot(
                messages=messages,
                model=model,
                max_tokens=request.max_tokens or 1024,
                system=system,
                stream=False
            )
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sophie-voice-bridge"}


@app.get("/")
async def root():
    return {
        "service": "Sophie Voice Bridge",
        "description": "OpenAI-compatible endpoint for ElevenLabs Custom LLM",
        "endpoint": "/v1/chat/completions",
        "docs": "/docs",
        "backend": CLAWDBOT_GATEWAY_URL
    }


if __name__ == "__main__":
    import uvicorn
    print("🎙️ Sophie Voice Bridge starting on http://0.0.0.0:8013")
    print(f"📡 Backend: {CLAWDBOT_GATEWAY_URL}")
    print("📌 Point ElevenLabs Custom LLM to: http://<your-domain>:8013/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8013)
