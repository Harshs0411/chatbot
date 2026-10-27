"""
frontend.py: Streamlit User Interface for LangGraph Conversational Agent & RAG
Features:
- Multi-session chat threads with persistent history via SQLite checkpointer.
- Per-thread PDF document upload & indexing into ChromaDB (PyPDFLoader, no OCR).
- Real-time streaming with dynamic tool execution indicators.
- Automatic conversation title generation.
- Clean display of source citations when RAG retrieval is triggered.
"""

import os
import sys
import json
import uuid
from dotenv import load_dotenv

# Ensure project root directory is on sys.path regardless of execution directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv()

import streamlit as st

# Sync Streamlit Cloud Secrets (st.secrets) to environment variables if running in Streamlit Cloud
try:
    for secret_key, secret_val in st.secrets.items():
        if isinstance(secret_val, str):
            os.environ.setdefault(secret_key, secret_val)
except Exception:
    pass

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


from backend.backend import (
        workflow,
        ingest_pdf,
        thread_document_metadata,
        create_thread,
        get_all_threads,
        get_thread_title,
        update_thread_title,
        delete_thread,
        get_chat_model,
        get_openai_api_key,
)
# except ImportError:
#     from Backend.backend import (
#         workflow,
#         ingest_pdf,
#         thread_document_metadata,
#         create_thread,
#         get_all_threads,
#         get_thread_title,
#         update_thread_title,
#         delete_thread,
#         get_chat_model,
#     )

try:
    from evaluator import evaluate_rag_turn
except ImportError:
    evaluate_rag_turn = None

# Page configuration
st.set_page_config(
    page_title="AI Agent with RAG & Tools",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Helper Classes & Utilities
# ---------------------------------------------------------------------------
class ConversationTitle(BaseModel):
    name: str = Field(description="A concise descriptive title of 3-5 words summarizing the conversation topic.")


title_parser = PydanticOutputParser(pydantic_object=ConversationTitle)
title_model = get_chat_model()


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def generate_name(thread_id: str) -> str:
    """Generates a concise 3-5 word conversation title from early conversation messages."""
    try:
        state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        if not messages:
            return "New Chat"

        conversation_text = "\n".join(
            f"{m.type}: {m.content}" for m in messages[:4] if getattr(m, "content", None)
        )

        prompt = PromptTemplate(
            template="""You are a helpful naming assistant.
Generate a concise, descriptive title for this conversation.

Rules:
- Maximum 5 words
- No quotes or punctuation
- Summarize the main intent

{format_instructions}

Conversation:
{conversation_text}
""",
            input_variables=["conversation_text"],
            partial_variables={"format_instructions": title_parser.get_format_instructions()},
        )

        chain = prompt | title_model | title_parser
        result = chain.invoke({"conversation_text": conversation_text})
        return result.name.strip().strip('"').strip("'")
    except Exception:
        return "Chat Conversation"


def load_conversation_messages(thread_id: str):
    """Retrieves only user and assistant messages for UI display, filtering out internal tool payloads."""
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    raw_messages = state.values.get("messages", [])
    display_messages = []

    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            display_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            display_messages.append({"role": "assistant", "content": msg.content})

    return display_messages


def start_new_chat():
    """Initializes a fresh conversation thread."""
    new_id = generate_thread_id()
    create_thread(new_id, title="New Chat")
    st.session_state["current_thread"] = new_id
    st.session_state["message_history"] = []
    st.rerun()


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "current_thread" not in st.session_state:
    all_existing_threads = get_all_threads()
    if all_existing_threads:
        st.session_state["current_thread"] = all_existing_threads[0][0]
    else:
        new_id = generate_thread_id()
        create_thread(new_id, title="New Chat")
        st.session_state["current_thread"] = new_id

current_thread_id = str(st.session_state["current_thread"])

if "message_history" not in st.session_state or not st.session_state["message_history"]:
    st.session_state["message_history"] = load_conversation_messages(current_thread_id)

current_title = get_thread_title(current_thread_id) or "Chat"

# ---------------------------------------------------------------------------
# Sidebar UI: Thread Management & PDF Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 LangGraph Agent")
    st.caption("Conversational Memory • Thread-Scoped RAG • ReAct Tools")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()

    st.markdown("---")

    # Document Upload Section
    st.subheader("📄 Document Ingestion (RAG)")
    st.caption("Upload a digital PDF for this specific chat thread. (Fast native text extraction, zero OCR overhead)")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"uploader_{current_thread_id}",
        label_visibility="collapsed"
    )

    if uploaded_pdf:
        thread_docs = thread_document_metadata(current_thread_id)
        already_indexed = any(doc.get("filename") == uploaded_pdf.name for doc in thread_docs)

        if already_indexed:
            st.info(f"'{uploaded_pdf.name}' is already indexed for this chat.")
        else:
            with st.status("Indexing document into ChromaDB...", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=current_thread_id,
                        filename=uploaded_pdf.name,
                    )
                    status_box.update(
                        label=f" Indexed '{summary['filename']}' ({summary['pages']} pages, {summary['chunks']} chunks)",
                        state="complete",
                        expanded=False
                    )
                    st.rerun()
                except Exception as e:
                    status_box.update(label=f"Ingestion failed: {e}", state="error")

    # Display Documents Linked to Current Thread
    active_docs = thread_document_metadata(current_thread_id)
    if active_docs:
        with st.expander(f"📚 Indexed Documents ({len(active_docs)})", expanded=False):
            for doc in active_docs:
                st.write(f"- **{doc['filename']}** ({doc['chunks']} chunks)")
    else:
        st.caption("_No PDF attached to this thread yet._")

    st.markdown("---")

    # Conversation History List
    st.subheader("💬 Conversations")
    all_threads = get_all_threads()

    for t_id, title in all_threads:
        col1, col2 = st.columns([0.82, 0.18])
        is_active = (t_id == current_thread_id)
        button_label = f"👉 {title}" if is_active else title

        with col1:
            if st.button(button_label, key=f"thread_{t_id}", use_container_width=True):
                st.session_state["current_thread"] = t_id
                st.session_state["message_history"] = load_conversation_messages(t_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{t_id}", help="Delete chat"):
                delete_thread(t_id)
                # If deleted active chat, reset to another or create new
                if t_id == current_thread_id:
                    remaining = [t[0] for t in get_all_threads() if t[0] != t_id]
                    if remaining:
                        st.session_state["current_thread"] = remaining[0]
                        st.session_state["message_history"] = load_conversation_messages(remaining[0])
                st.rerun()

    st.markdown("---")

    # Model Context Protocol (MCP) Section
    st.subheader("🔌 Model Context Protocol (MCP)")
    st.caption("Standardized tool & resource interface for Claude Desktop, Cursor & Agent networks.")

    with st.expander("🛠️ MCP Server & Client Config", expanded=False):
        st.success("🟢 MCP Server: Active (stdio / sse)")
        st.markdown("**Tools:** `chat_agent_turn`, `search_rag_context`, `ingest_pdf_document`, `calculate_expression`, `evaluate_rag`, `list_conversation_threads`")
        st.markdown("**Resources:** `rag://threads`, `rag://system/status`, `rag://threads/{id}/documents`")

        st.markdown("#### Connect Claude Desktop")
        st.caption("Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:")
        claude_config = {
            "mcpServers": {
                "langgraph-rag-agent": {
                    "command": sys.executable,
                    "args": [os.path.abspath(os.path.join(ROOT_DIR, "mcp_server.py"))]
                }
            }
        }
        st.code(json.dumps(claude_config, indent=2), language="json")

        st.markdown("#### Connect Cursor")
        st.caption("Add this to `.cursor/mcp.json`:")
        cursor_config = {
            "mcpServers": {
                "rag-agent": {
                    "command": sys.executable,
                    "args": [os.path.abspath(os.path.join(ROOT_DIR, "mcp_server.py"))]
                }
            }
        }
        st.code(json.dumps(cursor_config, indent=2), language="json")

        try:
            from backend.mcp_client import get_mcp_client_status
            client_status = get_mcp_client_status()
            if client_status.get("config_found"):
                connected = client_status.get("connected_servers", [])
                server_names = [s.get("name") for s in connected if isinstance(s, dict)]
                total_tools = client_status.get("total_tools", 0)
                if connected:
                    st.info(f"Connected to {len(connected)} external MCP server(s) ({total_tools} dynamic tools loaded): {', '.join(server_names)}")
                else:
                    st.caption("External MCP servers configured but none currently connected.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown(f"### {current_title}")

# Render existing messages
for i, message in enumerate(st.session_state["message_history"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render Verified Source Citations if present
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"📚 Verified Sources & Citations ({len(sources)} passages)", expanded=False):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"**Source {idx}:** `{src.get('filename')}` &bull; **Page {src.get('page', 1)}**")
                    if src.get("snippet"):
                        st.caption(f"> {src['snippet']}")

        # Render Interactive RAG Triad Evaluation if context is available
        context = message.get("context", [])
        if context and evaluate_rag_turn:
            eval_key = f"eval_{current_thread_id}_{i}"
            with st.expander("📊 Evaluate Answer Quality (RAG Triad Benchmark)", expanded=False):
                if eval_key not in st.session_state:
                    if st.button("🚀 Run RAG Triad Evaluation", key=f"btn_{eval_key}"):
                        with st.spinner("Evaluating Groundedness, Relevance & Context Precision..."):
                            res = evaluate_rag_turn(
                                question=message.get("query", ""),
                                context=context,
                                answer=message.get("content", "")
                            )
                            st.session_state[eval_key] = res
                            st.rerun()
                else:
                    res = st.session_state[eval_key]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🎯 Faithfulness", f"{res.get('faithfulness_score', 0):.1f} / 5.0")
                    c2.metric("💡 Relevance", f"{res.get('answer_relevance_score', 0):.1f} / 5.0")
                    c3.metric("🔍 Precision", f"{res.get('context_precision_score', 0):.1f} / 5.0")
                    st.markdown(f"**Verdict:** `{res.get('verdict', 'N/A')}` &bull; Overall Score: **{res.get('overall_score', 0):.2f} / 5.0**")
                    st.markdown(f"- **Faithfulness (Groundedness):** {res.get('faithfulness_reasoning', '')}")
                    st.markdown(f"- **Answer Relevance:** {res.get('answer_relevance_reasoning', '')}")
                    st.markdown(f"- **Context Precision:** {res.get('context_precision_reasoning', '')}")

# Check for active OpenAI API key
active_key = get_openai_api_key()
if not active_key:
    st.warning(
        "🔑 **OpenAI API Key Not Found**\n\n"
        "To activate the agent:\n"
        "- **In Streamlit Cloud**: Click **Manage app** (bottom-right) &rarr; **Settings** &rarr; **Secrets** &rarr; add `OPENAI_API_KEY = \"sk-...\"`\n"
        "- **Or enter it below for this browser session:**"
    )
    temp_key = st.text_input("Enter OpenAI API Key:", type="password", key="session_key_entry")
    if temp_key and temp_key.strip():
        os.environ["OPENAI_API_KEY"] = temp_key.strip()
        st.success("API Key loaded successfully!")
        st.rerun()

# User Input Box
user_prompt = st.chat_input("Ask a question, request calculations, or inquire about uploaded PDFs...")

if user_prompt:
    if not get_openai_api_key():
        st.error("⚠️ Please configure your OPENAI_API_KEY in Streamlit Secrets or in the box above to run queries.")
        st.stop()

    # Append and render user message immediately
    st.session_state["message_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Runtime configuration passing the active thread_id into LangGraph
    runtime_config = {
        "configurable": {"thread_id": current_thread_id},
        "metadata": {"thread_id": current_thread_id},
        "run_name": "agent_execution_turn",
    }

    with st.chat_message("assistant"):
        status_container = {"box": None}
        captured_data = {"sources": [], "context": []}

        def stream_agent_response():
            full_content = ""
            for message_chunk, _ in workflow.stream(
                {"messages": [HumanMessage(content=user_prompt)]},
                config=runtime_config,
                stream_mode="messages",
            ):
                # Check for tool invocations
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_container["box"] is None:
                        if tool_name == "rag_tool":
                            label = "📚 Searching uploaded document..."
                        elif tool_name == "web_search":
                            label = "🌐 Searching web via DuckDuckGo (Corrective Fallback)..."
                        elif tool_name == "calculator_tool":
                            label = "🧮 Calculating expression..."
                        else:
                            label = f"⚙️ Executing {tool_name}..."

                        status_container["box"] = st.status(label, expanded=False)
                    else:
                        status_container["box"].update(label="Action completed", state="complete")

                    # Extract RAG context and sources for inspection & evaluation
                    if tool_name == "rag_tool":
                        try:
                            raw = message_chunk.content
                            data = json.loads(raw) if isinstance(raw, str) else raw
                            if isinstance(data, dict):
                                captured_data["sources"] = data.get("sources", [])
                                captured_data["context"] = data.get("context", [])
                        except Exception:
                            pass

                # Stream out final AI message tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    full_content += str(message_chunk.content)
                    yield message_chunk.content

        # Stream the text to the UI
        final_answer = st.write_stream(stream_agent_response())

        if status_container["box"] is not None:
            status_container["box"].update(label="Tools executed successfully", state="complete", expanded=False)

    # Record assistant message into local session state with grounding metadata
    if final_answer:
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": final_answer,
            "sources": captured_data["sources"],
            "context": captured_data["context"],
            "query": user_prompt,
        })

    # Auto-generate meaningful title after the first turn of a "New Chat"
    if current_title == "New Chat" and len(st.session_state["message_history"]) >= 2:
        fresh_title = generate_name(current_thread_id)
        if fresh_title and fresh_title != "New Chat":
            update_thread_title(current_thread_id, fresh_title)

    st.rerun()
