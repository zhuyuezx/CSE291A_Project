"""
MCP Host for Ollama (Qwen 3.5) — connects to MCP servers, discovers tools,
and orchestrates the LLM ↔ tool calling loop.

Designed to work safely inside Jupyter notebooks by running the MCP event
loop on a dedicated background thread, so anyio cancel-scopes are never
crossed between cells.

Architecture:
    ┌─────────────┐     MCP Protocol      ┌──────────────────┐
    │  MCP Host    │ ◄──── stdio ────────► │  MCP Server(s)   │
    │  (this file) │                       │  (weather, etc.) │
    │              │     Ollama HTTP API   └──────────────────┘
    │  Qwen 3.5    │ ◄──── REST ────────►  Ollama @ :11434
    └─────────────┘

Usage (Jupyter-safe):
    host = MCPHost(model="qwen3.5:35b")
    host.start()
    host.connect_server("weather", "python", ["-m", "tools.mcp_weather_server"])
    reply = host.chat("What is the weather in Paris?")
    print(reply)
    host.stop()
"""

import asyncio
import json
import threading
import requests
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPHost:
    """
    Jupyter-safe MCP Host that:
      1. Runs the async MCP machinery on a dedicated background thread
      2. Connects to one or more MCP servers (via stdio)
      3. Discovers their tools
      4. Converts MCP tool schemas → Ollama-compatible tool schemas
      5. Sends user queries to Qwen 3.5 with tool definitions
      6. Executes tool calls via MCP, feeds results back to the LLM
    """

    def __init__(self, model: str = "qwen3.5:35b",
                 ollama_url: str = "http://localhost:11434",
                 verbose: bool = True):
        self.model = model
        self.ollama_url = ollama_url
        self.verbose = verbose

        # Background event-loop state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # MCP state (only touched from the background loop)
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[dict] = []
        self._tool_to_session: dict[str, ClientSession] = {}

    # ==================================================================
    # Background event-loop helpers
    # ==================================================================

    def _run_loop(self):
        """Entry-point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        """Submit a coroutine to the background loop and block until done."""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("MCPHost is not started. Call host.start() first.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()            # blocks the calling thread

    # ==================================================================
    # Public synchronous API
    # ==================================================================

    def start(self):
        """Start the background event-loop thread and init the exit stack."""
        if self._thread is not None and self._thread.is_alive():
            return  # already started
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait until the loop is actually running
        while self._loop is None or not self._loop.is_running():
            pass
        # Create the async exit stack on the background loop
        self._submit(self._async_init())
        if self.verbose:
            print("✅ MCPHost started (background event-loop running)")

    def stop(self):
        """Shut down all MCP connections and the background loop."""
        if self._loop is None:
            return
        try:
            self._submit(self._async_shutdown())
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Shutdown warning: {e}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._sessions.clear()
        self._tools.clear()
        self._tool_to_session.clear()
        if self.verbose:
            print("✅ MCPHost stopped")

    def connect_server(self, name: str, command: str,
                       args: list[str] | None = None,
                       env: dict | None = None):
        """Connect to an MCP server process via stdio (sync wrapper)."""
        self._submit(self._async_connect_server(name, command, args, env))

    def chat(self, user_message: str, system: str | None = None) -> str:
        """Run a full tool-calling loop (sync wrapper)."""
        return self._submit(self._async_chat(user_message, system))

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a single tool via MCP (sync wrapper)."""
        return self._submit(self._async_call_tool(tool_name, arguments))

    # ==================================================================
    # Async internals (run on the background loop)
    # ==================================================================

    async def _async_init(self):
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

    async def _async_shutdown(self):
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None

    async def _async_connect_server(self, name: str, command: str,
                                    args: list[str] | None = None,
                                    env: dict | None = None):
        """Connect to an MCP server process via stdio."""
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._sessions[name] = session

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            ollama_schema = self._mcp_to_ollama_schema(tool)
            self._tools.append(ollama_schema)
            self._tool_to_session[tool.name] = session
            if self.verbose:
                print(f"  Registered tool: {tool.name} (from '{name}')")

        if self.verbose:
            print(f"✅ Connected to MCP server '{name}' — {len(tools_result.tools)} tool(s)")

    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------

    def _mcp_to_ollama_schema(self, mcp_tool) -> dict:
        """Convert an MCP tool schema to Ollama/OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "parameters": mcp_tool.inputSchema if mcp_tool.inputSchema else {
                    "type": "object", "properties": {}
                },
            },
        }

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_ollama(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Call Ollama chat API (pure HTTP, no async needed)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()

    async def _async_call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool via its MCP session."""
        session = self._tool_to_session.get(tool_name)
        if not session:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result = await session.call_tool(tool_name, arguments)
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
            else:
                texts.append(str(block))
        return "\n".join(texts)

    # ------------------------------------------------------------------
    # Main chat loop
    # ------------------------------------------------------------------

    async def _async_chat(self, user_message: str, system: str | None = None) -> str:
        """
        Full tool-calling loop:
          1. User message + tool schemas → LLM
          2. If LLM returns tool_calls → execute via MCP → feed back
          3. Repeat until LLM gives a final text answer
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"👤 User: {user_message}")
            print(f"{'='*60}")

        max_rounds = 5  # safety limit
        for round_num in range(max_rounds):
            resp = self._call_ollama(messages, self._tools if self._tools else None)
            assistant_msg = resp["message"]

            # Print thinking if present (Qwen 3.5 feature)
            if self.verbose and assistant_msg.get("thinking"):
                print(f"\n💭 Thinking: {assistant_msg['thinking'][:300]}...")

            tool_calls = assistant_msg.get("tool_calls", [])

            # Fallback: parse tool call from content if tool_calls is empty
            if not tool_calls and assistant_msg.get("content", "").strip().startswith("{"):
                try:
                    parsed = json.loads(assistant_msg["content"])
                    if "name" in parsed and "arguments" in parsed:
                        tool_calls = [{"function": parsed}]
                except json.JSONDecodeError:
                    pass

            # No tool calls → final answer
            if not tool_calls:
                answer = assistant_msg.get("content", "")
                if self.verbose:
                    print(f"\n🤖 Answer: {answer}")
                return answer

            # Execute tool calls
            messages.append(assistant_msg)
            for tc in tool_calls:
                func_info = tc.get("function", tc)
                func_name = func_info["name"]
                func_args = func_info.get("arguments", {})

                if self.verbose:
                    print(f"\n🔧 Tool call [{round_num+1}]: {func_name}({json.dumps(func_args)})")

                result = await self._async_call_tool(func_name, func_args)

                if self.verbose:
                    print(f"📦 Result: {result}")

                messages.append({"role": "tool", "content": result})

        return "⚠️ Max tool-calling rounds reached."

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_to_session.keys())

    @property
    def tool_schemas(self) -> list[dict]:
        return list(self._tools)
