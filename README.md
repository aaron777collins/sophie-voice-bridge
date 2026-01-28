# Sophie Voice Bridge

ElevenLabs Custom LLM server that connects to Sophie AI via Clawdbot.

## Architecture

```
WhatsApp Call → ElevenLabs (STT/TTS) → This Bridge → Clawdbot → Haiku
```

- **Haiku** handles fast, voice-optimized responses
- **Sophie persona** baked into the system prompt
- Voice-optimized output (short, conversational, no formatting)

## Quick Start (Docker)

```bash
# Clone
git clone https://github.com/aaron777collins/sophie-voice-bridge.git
cd sophie-voice-bridge

# Configure
cp .env.example .env
# Edit .env with your Clawdbot gateway token

# Run
docker compose up -d

# Check health
curl http://localhost:8013/health
```

## Configuration

Create `.env` file:

```env
CLAWDBOT_GATEWAY_URL=http://host.docker.internal:18789
CLAWDBOT_GATEWAY_TOKEN=your-gateway-token
```

**Note:** `host.docker.internal` routes to the host machine where Clawdbot runs.

## ElevenLabs Setup

1. **Create Agent** in ElevenLabs Agents Platform
2. **Agent Settings → LLM → Custom LLM**
3. **URL:** `https://voice.aaroncollins.info/v1/chat/completions`
4. **Model:** anything (ignored)
5. **Enable "Custom LLM extra body":** ✓
6. **Token Limit:** 5000

Then connect WhatsApp via Twilio per ElevenLabs docs.

## API

### Endpoint

```
POST /v1/chat/completions
```

OpenAI-compatible chat completions (streaming and non-streaming).

### Health Check

```
GET /health
```

### Test

```bash
curl -X POST http://localhost:8013/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hey Sophie!"}], "stream": false}'
```

## Development

```bash
# Local setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run locally
python bridge.py
```

## How It Works

1. ElevenLabs sends transcribed speech to `/v1/chat/completions`
2. Bridge adds Sophie's voice persona as system prompt
3. Request forwarded to Clawdbot's HTTP API (uses Haiku model)
4. Response streamed back to ElevenLabs
5. ElevenLabs converts to speech

## Voice Persona

The bridge includes a voice-optimized system prompt:
- Keep responses SHORT (1-3 sentences)
- Natural speech patterns, not formal prose
- No markdown or formatting (it's audio)
- Conversational and warm

## License

MIT
