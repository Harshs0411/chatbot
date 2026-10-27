"""
backend.py: Production-Grade Conversational Agent with Thread-Isolated RAG
Built using LangGraph, ChromaDB, SQLite, and DuckDuckGo Search.

Architecture Highlights:
1. ReAct Cyclic Agent: START -> chatbot -> tools_condition -> tools -> chatbot -> END.
2. Thread-Isolated RAG: Ingests digital PDFs (PyPDFLoader, no OCR), chunks them, and
   restricts retrieval strictly to documents mapped to the active thread in SQLite.
3. Persistent State: Uses LangGraph's SqliteSaver checkpointer for multi-turn thread memory.
4. Clean Tool Runtime Injection: Extracts thread_id from RunnableConfig automatically.
"""

import os
import tempfile
import uuid
import sqlite3
from typing import TypedDict, Annotated, Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# 1. Environment & Model Configuration
# ---------------------------------------------------------------------------
load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Silence noisy Chroma telemetry logger due to posthog 3.x capture argument signature changes
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATABASE_NAME = os.path.join(BASE_DIR, "chatbot.db")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "chatbot_documents"


def get_chat_model():
    """
    Initializes the chat model.
    Prioritizes OpenAI if OPENAI_API_KEY is available;
    otherwise gracefully falls back to Ollama or local LLM.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=0.2,
        )
    except Exception:
        # Fallback to standard ChatOpenAI (will prompt for API key if missing)
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )


def get_embedding_model():
    """
    Initializes embedding model.
    Defaults to OpenAI text-embedding-3-small if key is present,
    otherwise falls back to OllamaEmbeddings.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    except Exception:
        return OpenAIEmbeddings(model="text-embedding-3-small")



# Global Vector Store
embedding_model = get_embedding_model()
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR,
)

model = get_chat_model()

# ---------------------------------------------------------------------------
# 2. SQLite Database & Persistence Helpers
# ---------------------------------------------------------------------------
def get_connection():
    """Returns a thread-safe SQLite connection."""
    return sqlite3.connect(DATABASE_NAME, check_same_thread=False)


def initialize_database():
    """Initializes tables for conversation threads, documents, and their mappings."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                documents_count INTEGER,
                chunks_count INTEGER,
                status TEXT DEFAULT 'READY',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_documents (
                thread_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (thread_id, document_id),
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


# Ensure DB schema is ready
initialize_database()


def create_thread(thread_id: str, title: str = "New Chat") -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO threads (thread_id, title) VALUES (?, ?)",
            (str(thread_id), title)
        )
        conn.commit()


def update_thread_title(thread_id: str, title: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE threads SET title = ? WHERE thread_id = ?",
            (title, str(thread_id))
        )
        conn.commit()


def get_thread_title(thread_id: str) -> Optional[str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title FROM threads WHERE thread_id = ?",
            (str(thread_id),)
        )
        row = cursor.fetchone()
        return row[0] if row else None


def get_all_threads() -> List[tuple]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT thread_id, title
            FROM threads
            ORDER BY created_at DESC
        """)
        return cursor.fetchall()


def delete_thread(thread_id: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM thread_documents WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM threads WHERE thread_id = ?", (str(thread_id),))
        conn.commit()


def create_document(document_id: str, filename: str, documents_count: int, chunks_count: int) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (document_id, filename, documents_count, chunks_count)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, filename, documents_count, chunks_count)
        )
        conn.commit()


def link_document_to_thread(thread_id: str, document_id: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO thread_documents (thread_id, document_id)
            VALUES (?, ?)
            """,
            (str(thread_id), str(document_id))
        )
        conn.commit()


def get_documents_for_thread(thread_id: str) -> List[tuple]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT d.document_id, d.filename, d.documents_count, d.chunks_count
            FROM documents d
            JOIN thread_documents td ON d.document_id = td.document_id
            WHERE td.thread_id = ?
            """,
            (str(thread_id),)
        )
        return cursor.fetchall()


def thread_document_metadata(thread_id: str) -> List[Dict[str, Any]]:
    docs = get_documents_for_thread(thread_id)
    return [
        {
            "document_id": row[0],
            "filename": row[1],
            "documents": row[2],
            "chunks": row[3],
        }
        for row in docs
    ]


# ---------------------------------------------------------------------------
# 3. Native PDF Ingestion (Fast, Deterministic)
# ---------------------------------------------------------------------------
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingests a digital PDF using PyPDFLoader (native text extraction).
    Splits text recursively with chunk overlap, attaches document metadata,
    indexes into ChromaDB, and links the document to the active thread in SQLite.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    document_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF using digital text stream parsing
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Recursive splitting preserves paragraph and sentence boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Tag each chunk with document and thread identifiers
        clean_filename = filename or os.path.basename(temp_path)
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = clean_filename
            # Preserve page number (1-indexed for human readability)
            if "page" in chunk.metadata:
                chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

        # Store in ChromaDB vector store
        vector_store.add_documents(chunks)

        # Track metadata in SQLite
        create_document(
            document_id=document_id,
            filename=clean_filename,
            documents_count=len(documents),
            chunks_count=len(chunks),
        )
        link_document_to_thread(
            thread_id=thread_id,
            document_id=document_id,
        )

        return {
            "document_id": document_id,
            "filename": clean_filename,
            "pages": len(documents),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 4. Agent Tools (RAG, Web Search, Arithmetic)
# ---------------------------------------------------------------------------
@tool
def calculator_tool(firstnum: float, secondnum: float, operation: str) -> dict:
    """
    Perform basic arithmetic calculations.
    Operations supported: 'add', 'sub', 'mul', 'div'.
    Example: calculator_tool(10, 5, 'add') -> 15
    """
    try:
        operation = operation.strip().lower()
        if operation == "add":
            result = firstnum + secondnum
        elif operation == "sub":
            result = firstnum - secondnum
        elif operation == "mul":
            result = firstnum * secondnum
        elif operation == "div":
            if secondnum == 0:
                return {"error": "Division by zero is undefined."}
            result = firstnum / secondnum
        else:
            return {"error": f"Unsupported operation '{operation}'. Supported: add, sub, mul, div."}

        return {
            "firstnum": firstnum,
            "secondnum": secondnum,
            "operation": operation,
            "result": round(result, 6)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF documents for the current chat thread.
    Use this tool whenever the user asks questions about their uploaded files, notes, or documents.
    """
    # Injected runtime config automatically provides the active thread_id
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    if not thread_id:
        return {"error": "Active thread ID is missing in runtime configuration."}

    thread_documents = get_documents_for_thread(thread_id)
    if not thread_documents:
        return {
            "status": "no_documents",
            "message": "No documents are currently indexed for this conversation. Prompt the user to upload a PDF in the sidebar if they wish to discuss specific files.",
            "query": query,
        }

    document_ids = [row[0] for row in thread_documents]

    # Scoped similarity search restricted strictly to this thread's documents
    relevant_docs = vector_store.similarity_search(
        query,
        k=4,
        filter={"document_id": {"$in": document_ids}}
    )

    if not relevant_docs:
        return {
            "status": "insufficient_context",
            "message": "Searched the uploaded document(s), but found NO relevant information or passages for this query. Use web_search as a fallback to look up the answer online.",
            "query": query,
            "context": [],
            "sources": []
        }

    context = [doc.page_content for doc in relevant_docs]
    sources = [
        {
            "filename": doc.metadata.get("filename", "document.pdf"),
            "page": doc.metadata.get("page", 1),
            "snippet": doc.page_content[:150].strip() + "..."
        }
        for doc in relevant_docs
    ]

    return {
        "status": "success",
        "query": query,
        "context": context,
        "sources": sources
    }


search_tool = DuckDuckGoSearchRun(name="web_search", region="us-en")

# Dynamic MCP Tools integration (Model Context Protocol)
try:
    from backend.mcp_client import get_dynamic_mcp_tools
    external_mcp_tools = get_dynamic_mcp_tools()
except Exception:
    external_mcp_tools = []

tools = [calculator_tool, search_tool, rag_tool] + external_mcp_tools
model_with_tools = model.bind_tools(tools=tools)


# ---------------------------------------------------------------------------
# 5. LangGraph State Machine Definition
# ---------------------------------------------------------------------------
class schema(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat(state: schema, config: RunnableConfig = None):
    """
    Core agent node that inspects conversational state,
    injects guidance prompt, and decides whether to respond directly or invoke a tool.
    Implements Corrective RAG (CRAG): if rag_tool yields insufficient context,
    the agent autonomously pivots to web_search to prevent hallucinations.
    """
    system_prompt = SystemMessage(
        content=(
            "You are an intelligent, helpful, and concise AI assistant with access to tools.\n"
            "- For questions about the user's uploaded PDF or study material, ALWAYS call `rag_tool` first.\n"
            "- CORRECTIVE RAG POLICY: If `rag_tool` returns 'insufficient_context' or no relevant passages, "
            "you must immediately call `web_search` as an autonomous fallback to answer the query, "
            "and clearly mention to the user that the information was retrieved via web search because the document lacked the answer.\n"
            "- For recent news, current facts, or general web inquiries, use `web_search`.\n"
            "- For calculations or arithmetic equations, use `calculator_tool`.\n"
            "- When answering based on retrieved document context, cite the filename and page number clearly.\n"
            "- If the user asks about a document but none has been uploaded to this thread, advise them to upload one in the sidebar."
        )
    )

    messages = [system_prompt, *state["messages"]]
    response = model_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


# Build the Graph
graph = StateGraph(schema)
graph.add_node("chatbot", chat)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

# Persistent checkpointing with SqliteSaver
checkpoint_conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_conn)
checkpointer.setup()

workflow = graph.compile(checkpointer=checkpointer)