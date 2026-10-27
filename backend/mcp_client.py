"""
mcp_client.py: Production-Grade Model Context Protocol (MCP) Client for LangGraph.

Architecture:
1. Multi-Server Connection Pool:
   - Supports persistent `stdio` (local subprocess) and `sse` (HTTP/streaming) transports.
   - Background event-loop worker thread per session to guarantee AnyIO/asyncio task safety.
2. Dynamic Schema Synthesis:
   - Parses MCP JSON Schema from `list_tools()`.
   - Dynamically constructs typed Pydantic v2 argument schemas using `create_model`.
   - Instantiates LangChain `StructuredTool` objects for seamless LangGraph binding.
3. Thread-Safe Lifecycle & Cache Management:
   - Thread-safe singleton manager (`MCPClientManager`).
   - Grouped tool cache by server with automatic cache rebuilding.
   - Graceful shutdown handlers and orphan process prevention (`atexit`).
   - Standard configuration discovery via `mcp_servers.json`.
"""

import atexit
import json
import logging
import os
import sys
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "mcp_servers.json")

# Self-healing interpreter auto-re-exec:
# If this file is directly executed under macOS system Python (/usr/bin/python3) or an interpreter
# without the 'mcp' package, automatically re-exec under the project's Anaconda environment.
try:
    import mcp
except ImportError:
    if __name__ == "__main__":
        for _candidate in ["/opt/anaconda3/bin/python", "/opt/anaconda3/bin/python3"]:
            if os.path.isfile(_candidate) and sys.executable != _candidate:
                os.execv(_candidate, [_candidate] + sys.argv)

# ================================================================
# JSON SCHEMA TO PYDANTIC V2 CONVERTER
# ================================================================

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def mcp_schema_to_pydantic(tool_name: str, schema: Optional[Dict[str, Any]]) -> type[BaseModel]:
    """
    Translates an MCP Tool's inputSchema (JSON Schema) into a dynamic Pydantic v2 BaseModel.
    Ensures LangChain and LangGraph receive accurate type constraints and descriptions.
    """
    if not schema or not isinstance(schema, dict):
        return create_model(f"{tool_name}_input")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: Dict[str, Tuple[Any, Any]] = {}

    for field_name, prop in properties.items():
        field_type = TYPE_MAP.get(prop.get("type"), Any)
        desc = prop.get("description", "")
        default = prop.get("default", ... if field_name in required else None)

        if field_name in required and default is ...:
            fields[field_name] = (field_type, Field(description=desc))
        else:
            fields[field_name] = (Optional[field_type], Field(default=default, description=desc))

    if not fields:
        return create_model(f"{tool_name}_input")

    return create_model(f"{tool_name}_input", **fields)


# ================================================================
# PERSISTENT ANYIO/ASYNCIO WORKER FOR MCP SESSIONS
# ================================================================

class MCPSessionWorker:
    """
    Manages a single active MCP client connection inside an isolated AnyIO/asyncio worker.
    Avoids AnyIO cancel-scope cross-task boundary issues by keeping stdio_client / sse_client
    and ClientSession inside a persistent task queue loop.
    """

    def __init__(
        self,
        name: str,
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env or os.environ.copy()
        self.env["MCP_SUBPROCESS"] = "1"
        self.cwd = cwd or ROOT_DIR
        self.url = url
        self.headers = headers

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"mcp-{name}-worker",
        )
        self.request_queue: Optional[asyncio.Queue] = None
        self.ready_event = threading.Event()
        self.init_error: Optional[Exception] = None
        self.is_connected = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._worker())

    async def _worker(self):
        self.request_queue = asyncio.Queue()
        try:
            from mcp.client.session import ClientSession

            if self.transport == "stdio":
                from mcp.client.stdio import stdio_client, StdioServerParameters

                params = StdioServerParameters(
                    command=self.command,
                    args=self.args,
                    env=self.env,
                    cwd=self.cwd,
                )
                stream_ctx = stdio_client(params)
            elif self.transport == "sse":
                from mcp.client.sse import sse_client

                stream_ctx = sse_client(self.url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported MCP transport: {self.transport}")

            async with stream_ctx as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.is_connected = True
                    self.ready_event.set()

                    while True:
                        req = await self.request_queue.get()
                        if req is None:  # Shutdown signal
                            break
                        action, args, future = req
                        try:
                            if action == "list_tools":
                                res = await session.list_tools()
                            elif action == "call_tool":
                                tool_name, tool_args = args
                                res = await session.call_tool(tool_name, tool_args)
                            elif action == "list_resources":
                                res = await session.list_resources()
                            elif action == "read_resource":
                                res = await session.read_resource(args[0])
                            elif action == "list_prompts":
                                res = await session.list_prompts()
                            elif action == "get_prompt":
                                prompt_name, prompt_args = args
                                res = await session.get_prompt(prompt_name, prompt_args)
                            else:
                                raise ValueError(f"Unknown MCP action: {action}")
                            future.set_result(res)
                        except Exception as ex:
                            future.set_exception(ex)
                        finally:
                            self.request_queue.task_done()
        except ImportError as e:
            if "mcp" in str(e).lower():
                self.init_error = ImportError(
                    f"The 'mcp' SDK is not installed in the currently active Python interpreter:\n"
                    f"  Active Interpreter: {sys.executable} (Python {sys.version.split()[0]})\n"
                    f"Please run using the project's Anaconda environment:\n"
                    f"  /opt/anaconda3/bin/python {sys.argv[0] if sys.argv else ''}\n"
                    f"or activate the conda environment: 'conda activate base'"
                )
            else:
                self.init_error = e
            self.is_connected = False
            self.ready_event.set()
        except Exception as e:
            self.init_error = e
            self.is_connected = False
            self.ready_event.set()

    def start(self, timeout: float = 20.0) -> List[Any]:
        """Starts the worker thread and initializes the remote MCP session."""
        self.thread.start()
        if not self.ready_event.wait(timeout=timeout):
            raise TimeoutError(f"Connection to MCP server '{self.name}' timed out after {timeout}s.")
        if self.init_error:
            raise self.init_error
        res = self._send_request("list_tools", ())
        return getattr(res, "tools", [])

    def _send_request(self, action: str, args: Tuple, timeout: float = 60.0) -> Any:
        if not self.is_connected or self.request_queue is None:
            raise RuntimeError(f"MCP server '{self.name}' is not connected.")

        future = self.loop.create_future()
        self.loop.call_soon_threadsafe(self.request_queue.put_nowait, (action, args, future))

        done_event = threading.Event()
        holder: Dict[str, Any] = {}

        def _cb(f):
            try:
                holder["res"] = f.result()
            except Exception as ex:
                holder["err"] = ex
            done_event.set()

        self.loop.call_soon_threadsafe(future.add_done_callback, _cb)

        if not done_event.wait(timeout=timeout):
            raise TimeoutError(f"MCP action '{action}' on server '{self.name}' timed out after {timeout}s.")

        if "err" in holder:
            raise holder["err"]
        return holder["res"]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Calls a tool synchronously through the active session."""
        raw_result = self._send_request("call_tool", (tool_name, arguments))
        # Format result into string for the LLM
        texts = []
        if hasattr(raw_result, "content") and raw_result.content:
            for block in raw_result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                else:
                    texts.append(str(block))
        if hasattr(raw_result, "structured_content") and raw_result.structured_content and not texts:
            return json.dumps(raw_result.structured_content, indent=2)

        output = "\n".join(texts)
        if getattr(raw_result, "is_error", False):
            return f"[MCP Error] Tool '{tool_name}' failed: {output}"
        return output or "Tool executed successfully (no output)."

    def list_resources(self) -> List[Any]:
        res = self._send_request("list_resources", ())
        return getattr(res, "resources", [])

    def read_resource(self, uri: str) -> str:
        raw_result = self._send_request("read_resource", (uri,))
        if hasattr(raw_result, "contents") and raw_result.contents:
            return getattr(raw_result.contents[0], "text", str(raw_result.contents[0]))
        return str(raw_result)

    def close(self):
        """Cleanly terminates the connection and background thread."""
        self.is_connected = False
        if self.request_queue and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.request_queue.put_nowait, None)
        self.thread.join(timeout=3.0)


# ================================================================
# CENTRAL MCP CLIENT MANAGER
# ================================================================

class MCPClientManager:
    """
    Central manager for all MCP server connections.

    Responsibilities:
    - Maintain one MCPSessionWorker per MCP server
    - Load MCP server configuration
    - Discover and convert MCP tools into LangChain StructuredTools
    - Cache discovered tools
    - Provide connection/status information
    - Cleanly shut down all MCP workers
    """

    _instance: Optional["MCPClientManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.sessions: Dict[str, MCPSessionWorker] = {}

        # Tools grouped by server for clean cache management.
        self.tools_by_server: Dict[str, List[BaseTool]] = {}

        # Flat list used by LangGraph/LangChain.
        self.cached_tools: List[BaseTool] = []

        # Protects sessions/cache from concurrent access.
        self._lock = threading.RLock()

        self._closed = False

        # Backup cleanup when Python exits.
        atexit.register(self.close_all)

    # ============================================================
    # SINGLETON
    # ============================================================

    @classmethod
    def get_instance(cls) -> "MCPClientManager":
        """
        Returns the single MCPClientManager instance.
        Thread-safe singleton initialization.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()

        return cls._instance

    # ============================================================
    # CONNECT SERVER
    # ============================================================

    def connect_server(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[BaseTool]:
        """
        Connect to one MCP server and return its LangChain tools.
        """
        if self._closed:
            raise RuntimeError("MCPClientManager has been closed.")

        with self._lock:
            existing_worker = self.sessions.get(name)
            if existing_worker and existing_worker.is_connected:
                logger.debug("MCP server '%s' is already connected.", name)
                return list(self.tools_by_server.get(name, []))

        transport = "sse" if url else "stdio"
        logger.info("Connecting to MCP server '%s' using %s transport.", name, transport)

        worker = MCPSessionWorker(
            name=name,
            transport=transport,
            command=command,
            args=args or [],
            env=env,
            cwd=cwd,
            url=url,
            headers=headers,
        )

        try:
            mcp_tools = worker.start()

            converted_tools = self._convert_tools(
                server_name=name,
                worker=worker,
                mcp_tools=mcp_tools,
            )

            with self._lock:
                self.sessions[name] = worker
                self.tools_by_server[name] = converted_tools
                self._rebuild_tool_cache()

            logger.info("Connected to MCP server '%s'. Loaded %d tools.", name, len(converted_tools))
            return list(converted_tools)

        except Exception:
            logger.exception("Failed to connect to MCP server '%s'.", name)
            try:
                worker.close()
            except Exception:
                logger.exception("Failed to close partially initialized worker '%s'.", name)
            return []

    # ============================================================
    # TOOL CONVERSION
    # ============================================================

    def _convert_tools(
        self,
        server_name: str,
        worker: MCPSessionWorker,
        mcp_tools: List[Any],
    ) -> List[BaseTool]:
        """
        Convert raw MCP tools into LangChain StructuredTools.
        Prefixed with server name for deterministic namespace isolation.
        """
        langchain_tools: List[BaseTool] = []

        for tool in mcp_tools:
            try:
                original_name = getattr(tool, "name", None)
                if not original_name:
                    logger.warning("Skipping MCP tool from '%s' because it has no name.", server_name)
                    continue

                tool_name = f"{server_name}_{original_name}"
                description = getattr(tool, "description", None) or f"MCP tool '{original_name}' from server '{server_name}'"

                raw_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
                schema_model = mcp_schema_to_pydantic(original_name, raw_schema)

                executor = self._make_executor(
                    worker=worker,
                    target_tool_name=original_name,
                )

                structured_tool = StructuredTool.from_function(
                    name=tool_name,
                    description=description,
                    args_schema=schema_model,
                    func=executor,
                )
                langchain_tools.append(structured_tool)

            except Exception:
                logger.exception("Failed to convert MCP tool '%s' from server '%s'.", getattr(tool, "name", "<unknown>"), server_name)

        return langchain_tools

    @staticmethod
    def _make_executor(worker: MCPSessionWorker, target_tool_name: str):
        """Creates a callable forwarding execution to the active MCP worker."""
        def execute(**kwargs) -> str:
            if not worker.is_connected:
                raise RuntimeError(f"MCP server '{worker.name}' is not connected.")
            return worker.call_tool(target_tool_name, kwargs)
        return execute

    def _rebuild_tool_cache(self) -> None:
        """Rebuild flat tool cache from tools_by_server."""
        all_tools: List[BaseTool] = []
        for tools in self.tools_by_server.values():
            all_tools.extend(tools)
        self.cached_tools = all_tools

    # ============================================================
    # LOAD CONFIG
    # ============================================================

    def load_from_config(self, config_path: str = CONFIG_PATH) -> List[BaseTool]:
        """
        Load MCP server definitions from mcp_servers.json.
        Connects to configured servers and removes decommissioned ones.
        """
        if self._closed:
            raise RuntimeError("MCPClientManager has been closed.")

        if not os.path.isfile(config_path):
            return []

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.exception("Could not read MCP configuration: %s", e)
            return []

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            logger.error("'mcpServers' must be a JSON object.")
            return []

        configured_server_names = set(servers.keys())

        with self._lock:
            running_server_names = set(self.sessions.keys())

        removed_servers = running_server_names - configured_server_names
        for server_name in removed_servers:
            logger.info("Server '%s' removed from configuration. Closing connection.", server_name)
            self.disconnect_server(server_name)

        all_tools: List[BaseTool] = []
        for server_name, server_config in servers.items():
            if not isinstance(server_config, dict):
                continue

            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env")
            cwd = server_config.get("cwd") or ROOT_DIR
            url = server_config.get("url")
            headers = server_config.get("headers")

            # Environment & portability normalization:
            # If command is relative or points to a missing interpreter path, use current sys.executable
            if command:
                if not os.path.isabs(command) or not os.path.exists(command):
                    command = sys.executable
            if args and isinstance(args, list):
                resolved_args = []
                for a in args:
                    if isinstance(a, str) and a.endswith(".py") and not os.path.exists(a):
                        candidate = os.path.join(ROOT_DIR, os.path.basename(a))
                        if os.path.exists(candidate):
                            resolved_args.append(candidate)
                        else:
                            resolved_args.append(a)
                    else:
                        resolved_args.append(a)
                args = resolved_args

            if url and command:
                logger.error("Server '%s' specifies both URL and command.", server_name)
                continue
            if not url and not command:
                logger.error("Server '%s' has neither URL nor command.", server_name)
                continue

            tools = self.connect_server(
                name=server_name,
                command=command,
                args=args,
                env=env,
                cwd=cwd,
                url=url,
                headers=headers,
            )
            all_tools.extend(tools)

        with self._lock:
            self._rebuild_tool_cache()
            return list(self.cached_tools)

    def get_tools_for_server(self, name: str, refresh: bool = False) -> List[BaseTool]:
        """Return tools belonging to one MCP server."""
        with self._lock:
            worker = self.sessions.get(name)
            if not worker or not worker.is_connected:
                return []
            if not refresh:
                return list(self.tools_by_server.get(name, []))

        try:
            mcp_result = worker._send_request("list_tools", ())
            raw_tools = getattr(mcp_result, "tools", [])
            converted_tools = self._convert_tools(server_name=name, worker=worker, mcp_tools=raw_tools)
            with self._lock:
                self.tools_by_server[name] = converted_tools
                self._rebuild_tool_cache()
            return list(converted_tools)
        except Exception:
            logger.exception("Failed to refresh tools for MCP server '%s'.", name)
            with self._lock:
                return list(self.tools_by_server.get(name, []))

    def get_all_tools(self) -> List[BaseTool]:
        """Return all currently cached active tools."""
        with self._lock:
            return list(self.cached_tools)

    def disconnect_server(self, name: str) -> None:
        """Close and remove one MCP server connection."""
        with self._lock:
            worker = self.sessions.pop(name, None)
            self.tools_by_server.pop(name, None)
            self._rebuild_tool_cache()

        if worker is None:
            return

        try:
            worker.close()
            logger.info("MCP server '%s' disconnected.", name)
        except Exception:
            logger.exception("Error while closing MCP server '%s'.", name)

    def get_status(self) -> Dict[str, Any]:
        """Return current MCP connection status."""
        with self._lock:
            connected_servers = []
            for name, worker in self.sessions.items():
                if not worker.is_connected:
                    continue
                tools = self.tools_by_server.get(name, [])
                connected_servers.append({
                    "name": name,
                    "transport": worker.transport,
                    "tools_count": len(tools),
                })
            total_tools = sum(len(tools) for tools in self.tools_by_server.values())
            return {
                "config_found": os.path.isfile(CONFIG_PATH),
                "connected_servers": connected_servers,
                "total_tools": total_tools,
                "manager_closed": self._closed,
            }

    def close_all(self) -> None:
        """Gracefully close all MCP server workers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = list(self.sessions.items())
            self.sessions.clear()
            self.tools_by_server.clear()
            self.cached_tools.clear()

        for name, worker in workers:
            try:
                worker.close()
                logger.info("Closed MCP server '%s'.", name)
            except Exception:
                logger.exception("Failed to close MCP server '%s'.", name)

        logger.info("All MCP server connections closed.")


# ================================================================
# PUBLIC APPLICATION API
# ================================================================

def get_dynamic_mcp_tools() -> List[BaseTool]:
    """
    Public entrypoint for LangGraph.
    Loads MCP servers from mcp_servers.json and returns LangChain-compatible tools.
    """
    if os.getenv("MCP_SUBPROCESS") == "1":
        return []
    manager = MCPClientManager.get_instance()
    return manager.load_from_config()


def get_mcp_client_status() -> Dict[str, Any]:
    """
    Public entrypoint for Streamlit/UI.
    Returns the current cached connection status.
    """
    manager = MCPClientManager.get_instance()
    return manager.get_status()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print("=== MCP Client Inspection ===")
    manager = MCPClientManager.get_instance()
    tools = get_dynamic_mcp_tools()
    print(f"\nLoaded {len(tools)} tools from mcp_servers.json\n")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    status = get_mcp_client_status()
    print("\nStatus:", json.dumps(status, indent=2))
    manager.close_all()