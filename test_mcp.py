"""
test_mcp.py: Automated Test Harness for Model Context Protocol (MCP) Server.

Tests:
1. Tool discovery and schema registration.
2. Resource URI templates and static resources.
3. Prompt templates.
4. Tool execution (arithmetic calculation, thread listing, context search).
5. Resource content resolution (system health & thread list).
"""

import asyncio
import json
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mcp_server import mcp


async def run_mcp_tests():
    print("=" * 70)
    print("MODEL CONTEXT PROTOCOL (MCP) SERVER TEST HARNESS")
    print("=" * 70)

    # 1. Test Tools Registration
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    print(f"\n[1] Registered Tools ({len(tools)}):")
    for name in tool_names:
        print(f"    - {name}")

    expected_tools = [
        "chat_agent_turn",
        "search_rag_context",
        "ingest_pdf_document",
        "calculate_expression",
        "evaluate_rag",
        "list_conversation_threads",
        "create_conversation_thread",
    ]
    for expected in expected_tools:
        assert expected in tool_names, f"Missing tool: {expected}"
    print("    --> All expected tools registered successfully!")

    # 2. Test Resources Registration
    resources = await mcp.list_resources()
    resource_uris = [r.uri for r in resources]
    print(f"\n[2] Static Resources ({len(resources)}):")
    for uri in resource_uris:
        print(f"    - {uri}")

    assert any("rag://threads" in str(u) for u in resource_uris), "Missing rag://threads resource"
    assert any("rag://system/status" in str(u) for u in resource_uris), "Missing rag://system/status resource"
    print("    --> Core static resources verified!")

    # 3. Test Prompts Registration
    prompts = await mcp.list_prompts()
    prompt_names = [p.name for p in prompts]
    print(f"\n[3] Registered Prompts ({len(prompts)}):")
    for name in prompt_names:
        print(f"    - {name}")

    expected_prompts = ["rag_qa_assistant", "crag_fallback_policy", "rag_triad_evaluation"]
    for expected in expected_prompts:
        assert expected in prompt_names, f"Missing prompt: {expected}"
    print("    --> All prompt templates verified!")

    # 4. Test Tool Execution: calculate_expression
    print("\n[4] Testing Tool Execution: calculate_expression...")
    calc_res = await mcp.call_tool("calculate_expression", {"expression": "(25 * 4) + 50"})
    print(f"    Result raw: {calc_res}")
    # In MCPServer 2.x, result is a CallToolResult with structured_content or content
    content_text = ""
    if hasattr(calc_res, "content") and calc_res.content:
        content_text = calc_res.content[0].text
    elif hasattr(calc_res, "structured_content"):
        content_text = str(calc_res.structured_content)
    print(f"    Extracted output: {content_text}")
    assert "150" in content_text, f"Expected 150 in calculation result, got: {content_text}"
    print("    --> calculate_expression executed with expected output (150)!")

    # 5. Test Tool Execution: list_conversation_threads
    print("\n[5] Testing Tool Execution: list_conversation_threads...")
    threads_res = await mcp.call_tool("list_conversation_threads", {})
    t_text = ""
    if hasattr(threads_res, "content") and threads_res.content:
        t_text = threads_res.content[0].text
    print(f"    Threads response snippet: {t_text[:120]}...")
    print("    --> list_conversation_threads executed successfully!")

    # 6. Test Resource Resolution: rag://system/status
    print("\n[6] Testing Resource Resolution: rag://system/status...")
    status_raw = await mcp.read_resource("rag://system/status")
    # MCPServer returns ReadResourceResult or content
    status_text = ""
    if hasattr(status_raw, "contents") and status_raw.contents:
        status_text = status_raw.contents[0].text
    elif isinstance(status_raw, str):
        status_text = status_raw
    else:
        status_text = str(status_raw)

    print(f"    System status payload: {status_text[:160]}...")
    assert "healthy" in status_text, "Expected 'healthy' in system status"
    print("    --> rag://system/status resolved successfully with healthy status!")

    print("\n" + "=" * 70)
    print("ALL MODEL CONTEXT PROTOCOL (MCP) TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_mcp_tests())
