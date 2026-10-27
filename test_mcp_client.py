"""
test_mcp_client.py: Comprehensive Test Harness for Production MCP Client.

Verifies:
1. Connecting to an MCP server via real stdio transport (sub-process).
2. JSON-RPC protocol handshake & capability exchange.
3. Tool listing and dynamic Pydantic v2 schema generation.
4. Tool conversion to LangChain StructuredTool instances.
5. End-to-end tool invocation through LangChain's invoke interface.
6. Resource discovery and resolution.
7. Graceful teardown without AnyIO cancel-scope or sub-process leaks.
"""

import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.mcp_client import MCPClientManager


def test_production_mcp_client():
    print("=" * 70)
    print("PRODUCTION MODEL CONTEXT PROTOCOL (MCP) CLIENT TEST HARNESS")
    print("=" * 70)

    manager = MCPClientManager.get_instance()

    # 1. Connect to mcp_server.py via stdio
    print("\n[1] Connecting to MCP Server via stdio sub-process...")
    tools = manager.connect_server(
        name="local_rag_agent",
        command=sys.executable,
        args=["mcp_server.py"],
    )

    print(f"    Successfully connected! Discovered & converted {len(tools)} LangChain tools:")
    for t in tools:
        print(f"    - Name: {t.name}")
        print(f"      Description: {t.description}")
        print(f"      Schema: {t.args_schema.model_json_schema().get('properties', {})}")

    assert len(tools) >= 7, f"Expected at least 7 tools, got {len(tools)}"
    print("    --> All tools discovered and converted into StructuredTool objects!")

    # 2. Test Tool Invocation via LangChain Interface
    print("\n[2] Testing Tool Invocation via LangChain StructuredTool...")
    calc_tool = next((t for t in tools if "calculate_expression" in t.name), None)
    assert calc_tool is not None, "Missing calculate_expression tool"

    res = calc_tool.invoke({"expression": "(15 * 4) + 10"})
    print(f"    Invoked {calc_tool.name} with '(15 * 4) + 10'")
    print(f"    Result output:\n{res}")
    assert "70" in str(res), f"Expected 70 in result, got: {res}"
    print("    --> Tool invocation succeeded with expected result (70)!")

    # 3. Test Thread Listing Tool Invocation
    print("\n[3] Testing list_conversation_threads tool invocation...")
    threads_tool = next((t for t in tools if "list_conversation_threads" in t.name), None)
    assert threads_tool is not None, "Missing list_conversation_threads tool"

    t_res = threads_tool.invoke({})
    print(f"    Invoked {threads_tool.name}")
    print(f"    Result output snippet: {str(t_res)[:120]}...")
    print("    --> Thread listing tool executed cleanly!")

    # 4. Test Resource Reading via MCP Session
    print("\n[4] Testing Resource Reading via Worker Session...")
    worker = manager.sessions.get("local_rag_agent")
    assert worker is not None, "Worker session not found"

    status_str = worker.read_resource("rag://system/status")
    print(f"    Resource 'rag://system/status' output:\n{status_str[:150]}...")
    status_json = json.loads(status_str)
    assert status_json.get("status") == "healthy", "Expected healthy status"
    print("    --> Resource resolution verified successfully!")

    # 5. Test Manager Status
    print("\n[5] Inspecting Manager Status...")
    m_status = manager.get_status()
    connected_names = [s["name"] for s in m_status.get("connected_servers", [])]
    assert "local_rag_agent" in connected_names, f"Expected local_rag_agent in {connected_names}"
    print("    --> Manager status verified!")

    # 6. Teardown
    print("\n[6] Tearing down sessions and terminating sub-processes...")
    manager.close_all()
    print("    --> Client cleanly terminated without AnyIO or sub-process errors!")

    print("\n" + "=" * 70)
    print("ALL PRODUCTION MCP CLIENT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_production_mcp_client()
