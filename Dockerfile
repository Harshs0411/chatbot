# ==============================================================================
# Production Dockerfile for LangGraph Multi-Session RAG Agent & MCP Service
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure entrypoint has execution permissions
RUN chmod +x /app/entrypoint.sh

# Expose Streamlit (8501) and MCP SSE (8000) ports
EXPOSE 8501 8000

# Healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["streamlit"]
