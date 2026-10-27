#!/bin/bash
set -e

# Default to Streamlit UI if no arguments passed
MODE="${1:-streamlit}"

echo "=========================================================="
echo " Starting LangGraph Conversational Agent & RAG Service"
echo " Mode: ${MODE}"
echo "=========================================================="

# Check for API key warning
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARNING] OPENAI_API_KEY is not set. Ensure local Ollama or mock credentials are configured."
fi

# Ensure data directories exist
mkdir -p /app/chroma_db

if [ "$MODE" = "streamlit" ]; then
    echo "[*] Launching Streamlit Web UI on port ${STREAMLIT_SERVER_PORT:-8501}..."
    exec streamlit run frontend/frontend.py \
        --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
        --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
        --server.headless=true

elif [ "$MODE" = "mcp" ]; then
    echo "[*] Launching MCP Server (SSE transport) on port ${MCP_PORT:-8000}..."
    exec python mcp_server.py \
        --transport="${MCP_TRANSPORT:-sse}" \
        --port="${MCP_PORT:-8000}" \
        --host="${MCP_HOST:-0.0.0.0}"

elif [ "$MODE" = "all" ]; then
    echo "[*] Starting MCP Server in background..."
    python mcp_server.py --transport=sse --port=8000 --host=0.0.0.0 &
    MCP_PID=$!

    echo "[*] Starting Streamlit UI..."
    streamlit run frontend/frontend.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true &
    STREAMLIT_PID=$!

    # Trap termination signals and cleanly kill both child processes
    trap "kill $MCP_PID $STREAMLIT_PID; exit 0" SIGINT SIGTERM
    wait
else
    # Execute arbitrary command passed to docker run
    exec "$@"
fi
