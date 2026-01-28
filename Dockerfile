# Sophie Voice Bridge - ElevenLabs Custom LLM Server
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY bridge.py .

# Expose port
EXPOSE 8013

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8013/health || exit 1

# Run
CMD ["python", "bridge.py"]
