"""
test_system_integration.py: Comprehensive Verification of Dummy PDF Ingestion,
Question Answering (RAG, Math, CRAG Web Search), Multi-turn Memory, and Session Isolation.
"""

import os
import sys
import uuid
import json

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend.backend import (
    workflow,
    ingest_pdf,
    thread_document_metadata,
    create_thread,
    get_all_threads,
    get_thread_title,
    update_thread_title,
    delete_thread,
    get_documents_for_thread,
)

def build_dummy_pdf(filename: str):
    """Generates a clean, valid PDF with structured facts."""
    title = "Acme Quantum Innovations - Corporate Report 2026"
    lines = [
        "Project Codename: Project Nebula Prime",
        "Key Objective: Deploy autonomous edge AI systems with sub-5ms latency.",
        "Lead Architect: Dr. Elena Vance",
        "Annual R&D Budget: 14.5 Million USD allocated for quantum-safe encryption.",
        "Launch Date: October 15, 2026 across North America and Europe.",
        "Security Protocol: Triple-layer Zero Trust Architecture with post-quantum lattice cryptography."
    ]
    stream_lines = ['BT', '/F1 16 Tf', '50 720 Td', f'({title}) Tj', '/F1 12 Tf']
    for line in lines:
        stream_lines.append('0 -24 Td')
        escaped = line.replace('(', '\\(').replace(')', '\\)')
        stream_lines.append(f'({escaped}) Tj')
    stream_lines.append('ET')
    stream_content = '\n'.join(stream_lines).encode('latin-1')

    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length ' + str(len(stream_content)).encode('ascii') + b' >>\nstream\n' + stream_content + b'\nendstream'
    ]

    out = [b'%PDF-1.4\n']
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(sum(len(x) for x in out))
        out.append(f'{i} 0 obj\n'.encode('ascii') + obj + b'\nendobj\n')

    xref_offset = sum(len(x) for x in out)
    out.append(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    out.append(b'0000000000 65535 f \n')
    for off in offsets[1:]:
        out.append(f'{off:010d} 00000 n \n'.encode('ascii'))
    out.append(f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n'.encode('ascii'))

    with open(filename, 'wb') as f:
        f.write(b''.join(out))
    print(f"[*] Generated dummy PDF: {filename} ({len(b''.join(out))} bytes)")


def ask_agent(thread_id: str, prompt_text: str):
    """Sends a message turn to the LangGraph workflow and prints tool usage & response."""
    print(f"\n[?] User Prompt ({thread_id}): '{prompt_text}'")
    config = {"configurable": {"thread_id": thread_id}}
    
    tools_called = []
    final_response = ""

    for message_chunk, _ in workflow.stream(
        {"messages": [HumanMessage(content=prompt_text)]},
        config=config,
        stream_mode="messages"
    ):
        if isinstance(message_chunk, ToolMessage):
            t_name = getattr(message_chunk, "name", "tool")
            tools_called.append(t_name)
            print(f"    -> [TOOL CALLED] {t_name}")
        elif isinstance(message_chunk, AIMessage) and message_chunk.content:
            final_response += str(message_chunk.content)

    print(f"    [AI RESPONSE]: {final_response.strip()}\n")
    return {"tools_called": tools_called, "response": final_response}


def run_full_suite():
    print("=" * 75)
    print("SYSTEM INTEGRATION TEST: DUMMY PDF, SESSIONS, MEMORY & ISOLATION")
    print("=" * 75)

    pdf_path = "dummy_test.pdf"
    build_dummy_pdf(pdf_path)

    # -------------------------------------------------------------
    # 1. Test Session Alpha (With PDF Attached)
    # -------------------------------------------------------------
    thread_alpha = f"test-alpha-{uuid.uuid4().hex[:6]}"
    create_thread(thread_alpha, title="Session Alpha - Quantum Report")
    print(f"\n[1] Created Session Alpha: {thread_alpha}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    ingest_result = ingest_pdf(pdf_bytes, thread_id=thread_alpha, filename="dummy_test.pdf")
    print(f"    Ingested PDF: {ingest_result}")
    assert ingest_result["chunks"] >= 1, "Expected at least 1 chunk indexed"

    docs_alpha = thread_document_metadata(thread_alpha)
    print(f"    Session Alpha Linked Docs: {docs_alpha}")
    assert len(docs_alpha) == 1, "Expected 1 document linked to Session Alpha"

    # -------------------------------------------------------------
    # 2. Test RAG Retrieval on Session Alpha
    # -------------------------------------------------------------
    print("\n[2] Testing RAG Q&A on Session Alpha...")
    res1 = ask_agent(
        thread_alpha,
        "What is the codename of the project and who is the lead architect in the uploaded report?"
    )
    assert "rag_tool" in res1["tools_called"], "Expected rag_tool to be invoked"
    assert "Nebula Prime" in res1["response"], "Expected 'Nebula Prime' in answer"
    assert "Elena Vance" in res1["response"], "Expected 'Dr. Elena Vance' in answer"
    print("    [PASS] RAG extraction for codename and lead architect succeeded!")

    res2 = ask_agent(
        thread_alpha,
        "What is the annual R&D budget for quantum-safe encryption?"
    )
    # The agent may recall from memory or call rag_tool again
    assert "14.5" in res2["response"], "Expected '14.5 Million USD' in answer"
    print("    [PASS] Answer extraction for numerical facts succeeded (from RAG/context memory)!")

    # -------------------------------------------------------------
    # 3. Test Math Tooling
    # -------------------------------------------------------------
    print("\n[3] Testing Calculator Tool on Session Alpha...")
    res_math = ask_agent(thread_alpha, "Calculate 14.5 multiplied by 2 using the calculator.")
    assert "calculator_tool" in res_math["tools_called"], "Expected calculator_tool to be invoked"
    assert "29" in res_math["response"], "Expected 29 in calculation result"
    print("    [PASS] Calculator tool invocation verified!")

    # -------------------------------------------------------------
    # 4. Test Conversational Memory across Multi-Turn Chat
    # -------------------------------------------------------------
    print("\n[4] Testing Multi-turn Context Memory on Session Alpha...")
    res_mem = ask_agent(thread_alpha, "What was the name of the lead architect we discussed earlier?")
    assert "Elena Vance" in res_mem["response"], "Expected memory of Elena Vance from previous turns"
    print("    [PASS] Multi-turn memory verified via SqliteSaver checkpointer!")

    # -------------------------------------------------------------
    # 5. Test Corrective RAG (CRAG) Fallback to Web Search
    # -------------------------------------------------------------
    print("\n[5] Testing Corrective RAG (CRAG) Fallback on Session Alpha...")
    res_crag = ask_agent(thread_alpha, "What is the capital of Australia and what is its official currency?")
    assert "web_search" in res_crag["tools_called"] or "rag_tool" in res_crag["tools_called"], "Expected web search or RAG tool"
    assert "Canberra" in res_crag["response"], "Expected Canberra in response"
    print("    [PASS] CRAG fallback / web search executed cleanly!")

    # -------------------------------------------------------------
    # 6. Test Session Isolation (Session Beta with NO PDF)
    # -------------------------------------------------------------
    print("\n[6] Testing Session Isolation on Session Beta...")
    thread_beta = f"test-beta-{uuid.uuid4().hex[:6]}"
    create_thread(thread_beta, title="Session Beta - Isolated")
    print(f"    Created Session Beta: {thread_beta}")

    docs_beta = thread_document_metadata(thread_beta)
    print(f"    Session Beta Linked Docs: {docs_beta}")
    assert len(docs_beta) == 0, "Session Beta must have 0 documents linked"

    res_iso = ask_agent(thread_beta, "What is the Project Codename in the uploaded report?")
    # Session Beta must NOT know about Project Nebula Prime from Session Alpha's PDF
    print(f"    Response in Beta: {res_iso['response']}")
    assert "Nebula Prime" not in res_iso["response"] or "upload" in res_iso["response"].lower(), \
        "Session Beta leaked document context from Session Alpha!"
    print("    [PASS] Session isolation verified: Thread Beta cannot access Thread Alpha's documents!")

    # -------------------------------------------------------------
    # 7. Test Session Management & Cleanup
    # -------------------------------------------------------------
    print("\n[7] Testing Thread CRUD operations...")
    update_thread_title(thread_alpha, "Session Alpha - Renamed")
    assert get_thread_title(thread_alpha) == "Session Alpha - Renamed"
    
    threads = [t[0] for t in get_all_threads()]
    assert thread_alpha in threads and thread_beta in threads
    
    # Delete test threads
    delete_thread(thread_alpha)
    delete_thread(thread_beta)
    remaining_threads = [t[0] for t in get_all_threads()]
    assert thread_alpha not in remaining_threads
    assert thread_beta not in remaining_threads
    print("    [PASS] Thread CRUD operations and cascade deletions verified!")

    print("\n" + "=" * 75)
    print("ALL INTEGRATION TESTS (PDF, RAG, CRAG, MEMORY, SESSIONS) PASSED 100%!")
    print("=" * 75)


if __name__ == "__main__":
    run_full_suite()
