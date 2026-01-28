#!/usr/bin/env python3
"""
Sophie Voice Bridge - ElevenLabs Custom LLM Server

Architecture:
- Haiku handles fast voice responses (1-3 sentences, conversational)
- ask_sophie tool escalates to Sophie (Opus) for complex questions
- Sophie has full access to tools, memory, calendar, email, etc.

Usage:
    python bridge.py
    # Point ElevenLabs Custom LLM to https://voice.aaroncollins.info/v1/chat/completions
"""

import json
import os
import uuid
from typing import Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

# Config
CLAWDBOT_GATEWAY_URL = os.getenv("CLAWDBOT_GATEWAY_URL", "http://localhost:18789")
CLAWDBOT_GATEWAY_TOKEN = os.getenv("CLAWDBOT_GATEWAY_TOKEN", "")
VOICE_BRIDGE_API_KEY = os.getenv("VOICE_BRIDGE_API_KEY", "")

if not CLAWDBOT_GATEWAY_TOKEN:
    print("Warning: CLAWDBOT_GATEWAY_TOKEN not set - calls may fail")

if not VOICE_BRIDGE_API_KEY:
    print("⚠️  Warning: VOICE_BRIDGE_API_KEY not set - endpoint is UNPROTECTED!")

app = FastAPI(title="Sophie Voice Bridge")
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    authorization: str = Header(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
    api_key: str = Header(None, alias="api-key")
):
    """Verify the bearer token matches our API key."""
    if not VOICE_BRIDGE_API_KEY:
        # No key configured = open endpoint (dev mode)
        return True
    
    token = None
    
    # Try Bearer token from security scheme
    if credentials and credentials.credentials:
        token = credentials.credentials
    # Fallback: parse Authorization header directly
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    # Try X-API-Key header (common alternative)
    elif x_api_key:
        token = x_api_key
    # Try api-key header (OpenAI style)
    elif api_key:
        token = api_key
    
    # Debug logging
    print(f"[AUTH DEBUG] Authorization: {authorization[:30] if authorization else None}...")
    print(f"[AUTH DEBUG] X-API-Key: {x_api_key[:20] if x_api_key else None}...")
    print(f"[AUTH DEBUG] api-key: {api_key[:20] if api_key else None}...")
    print(f"[AUTH DEBUG] Token extracted: {token[:20] if token else None}...")
    
    if not token or token != VOICE_BRIDGE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

HAIKU_VOICE_SYSTEM = """You are Sophie's voice interface - a fast, conversational layer that handles quick questions and small talk.

VOICE GUIDELINES:
- Keep responses SHORT (1-3 sentences max)
- Speak naturally, not formally - this is audio, not text
- No markdown, bullet points, or formatting
- Use natural speech: "well", "so", "yeah" are fine

WHEN TO USE ask_sophie:
You have access to the ask_sophie tool which connects to Sophie's full brain (Opus model with tools).
Use it when Aaron asks about:
- His calendar, schedule, or upcoming events
- His emails or messages
- Files, projects, or code
- Research or complex technical questions
- Anything requiring memory of past conversations
- Tasks that need web search, browser, or other tools
- Business decisions or strategic thinking

For simple chat, greetings, quick math, or general knowledge - just answer directly.
When using ask_sophie, wait for the response and relay it conversationally.

Current time: {current_time}
"""

# Tool definitions (Anthropic format)
TOOLS = [
    {
        "name": "ask_sophie",
        "description": """Escalate to Sophie's full capabilities (Opus model with complete tool access).

USE THIS TOOL WHEN AARON ASKS ABOUT:
- Calendar, schedule, events, appointments
- Emails, messages, notifications  
- Files, documents, code, projects
- Research requiring web search or browsing
- Complex technical or business questions
- Anything requiring memory of past conversations
- Tasks needing tools (browser, file access, APIs)

Sophie can check his calendar, read emails, search the web, access files, and remember context from previous conversations.

DO NOT USE FOR:
- Simple greetings or small talk
- Basic math or general knowledge
- Questions you can confidently answer

When you get Sophie's response, relay it conversationally - don't just read it verbatim.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or request to send to Sophie. Include relevant context from the conversation."
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "How urgent is this request? High = needs immediate attention."
                }
            },
            "required": ["question"]
        }
    }
]

# ============================================================================
# MODELS
# ============================================================================

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

# ============================================================================
# CLAWDBOT API CALLS
# ============================================================================

async def call_haiku(messages: list, system: str, tools: list = None) -> dict:
    """Call Haiku via Clawdbot for fast voice responses."""
    
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    api_messages = [{"role": "system", "content": system}] + messages
    
    payload = {
        "model": "anthropic/claude-3-haiku-20240307",
        "messages": api_messages,
        "max_tokens": 1024,
        "stream": False
    }
    
    if tools:
        payload["tools"] = tools
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, 
                              detail=f"Haiku error: {response.text}")


async def call_sophie(question: str, urgency: str = "normal") -> str:
    """Call Sophie (Opus) via Clawdbot for complex questions requiring full capabilities."""
    
    headers = {
        "Authorization": f"Bearer {CLAWDBOT_GATEWAY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Build a prompt that tells Sophie this is from a voice call
    prompt = f"""[Voice Call Request - Urgency: {urgency}]

Aaron is on a WhatsApp voice call and asked: {question}

Please help with this request. Remember:
- Your response will be spoken aloud, so keep it conversational
- Be thorough but concise
- If you need to check calendar, emails, or use tools, do so
- Include key details Aaron needs to know"""
    
    payload = {
        "model": "anthropic/claude-opus-4-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for Opus + tools
        response = await client.post(
            f"{CLAWDBOT_GATEWAY_URL}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Sorry, I couldn't reach my full capabilities right now. Error: {response.status_code}"


def extract_text_content(response: dict) -> str:
    """Extract text content from OpenAI-style response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def extract_tool_calls(response: dict) -> list:
    """Extract tool calls from response if present."""
    try:
        return response["choices"][0]["message"].get("tool_calls", [])
    except (KeyError, IndexError):
        return []


async def stream_response(content: str, model: str = "claude-3-haiku-20240307"):
    """Convert a complete response into SSE streaming format for ElevenLabs."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(datetime.now().timestamp())
    
    # Stream the content in chunks (word by word for natural feel)
    words = content.split(' ')
    for i, word in enumerate(words):
        chunk_content = word + (' ' if i < len(words) - 1 else '')
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send finish chunk
    finish_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")  # ElevenLabs uses this path
async def create_chat_completion(
    request: ChatCompletionRequest,
    _auth: bool = Depends(verify_api_key)
):
    """OpenAI-compatible chat completions endpoint for ElevenLabs."""
    
    # Debug: log the incoming request
    print(f"[REQUEST] stream={request.stream}, model={request.model}")
    print(f"[REQUEST] messages={[m.role for m in request.messages]}")
    if request.elevenlabs_extra_body:
        print(f"[REQUEST] elevenlabs_extra_body={request.elevenlabs_extra_body}")
    
    system = HAIKU_VOICE_SYSTEM.format(
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    )
    
    # Convert messages
    messages = [{"role": m.role, "content": m.content} 
                for m in request.messages if m.role != "system"]
    
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    try:
        # First call to Haiku with tools
        haiku_response = await call_haiku(messages, system, TOOLS)
        
        # Check if Haiku wants to use a tool
        tool_calls = extract_tool_calls(haiku_response)
        
        if tool_calls:
            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == "ask_sophie":
                    args = json.loads(tool_call["function"]["arguments"])
                    question = args.get("question", "")
                    urgency = args.get("urgency", "normal")
                    
                    # Call Sophie
                    sophie_response = await call_sophie(question, urgency)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": sophie_response
                    })
            
            # Continue conversation with tool results
            messages.append(haiku_response["choices"][0]["message"])
            messages.extend(tool_results)
            
            # Get Haiku's final response after seeing Sophie's answer
            final_response = await call_haiku(messages, system)
        else:
            # No tool use - return Haiku's direct response
            final_response = haiku_response
        
    except Exception as e:
        # Return error in OpenAI format
        final_response = {
            "id": f"chatcmpl-error-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "claude-3-haiku-20240307",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Sorry, I ran into a problem: {str(e)}"
                },
                "finish_reason": "stop"
            }]
        }
    
    # If streaming requested, convert to SSE format
    if request.stream:
        content = extract_text_content(final_response)
        return StreamingResponse(
            stream_response(content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    return final_response


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sophie-voice-bridge"}


@app.get("/")
async def root():
    return {
        "service": "Sophie Voice Bridge",
        "description": "Haiku voice interface with Sophie (Opus) escalation",
        "endpoint": "/v1/chat/completions",
        "architecture": {
            "fast_layer": "Claude 3 Haiku - handles quick responses",
            "full_brain": "Claude Opus 4.5 (Sophie) - complex questions via ask_sophie tool"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    print("🎙️ Sophie Voice Bridge")
    print("   Fast layer: Haiku (1-3 sentence voice responses)")
    print("   Full brain: Sophie/Opus via ask_sophie tool")
    print(f"   Backend: {CLAWDBOT_GATEWAY_URL}")
    print("   Endpoint: http://0.0.0.0:8013/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8013)
