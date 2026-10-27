"""
backend package: Core LangGraph conversational agent and RAG pipeline.
"""
import sys
import os

from .backend import (
    workflow,
    ingest_pdf,
    thread_document_metadata,
    create_thread,
    get_all_threads,
    get_thread_title,
    update_thread_title,
    delete_thread,
    get_chat_model,
    get_embedding_model,
    vector_store,
    model,
    tools,
)

# Register 'Backend' alias in sys.modules for case-insensitive / backward compatibility
_current_module = sys.modules[__name__]
sys.modules.setdefault("Backend", _current_module)
if "backend.backend" in sys.modules:
    sys.modules.setdefault("Backend.backend", sys.modules["backend.backend"])

__all__ = [
    "workflow",
    "ingest_pdf",
    "thread_document_metadata",
    "create_thread",
    "get_all_threads",
    "get_thread_title",
    "update_thread_title",
    "delete_thread",
    "get_chat_model",
    "get_embedding_model",
    "vector_store",
    "model",
    "tools",
]
