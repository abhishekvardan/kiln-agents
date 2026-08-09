import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ._errors import KilnError
from .tool import Tool, ToolExecutionContext


def _extract_text(result: Any) -> str:
    return "\n".join(item.text for item in (result.content or []) if getattr(item, "type", None) == "text" and getattr(item, "text", None))


@dataclass
class MCPConnection:
    """Real kiln Tool objects, one per tool the server exposes — pass these straight into
    define_agent(tools=[...]); they compose with BoundAI.run()'s tool-calling loop exactly like
    hand-written tools."""

    tools: List[Tool]
    _stack: contextlib.AsyncExitStack

    async def close(self) -> None:
        """Terminates the server subprocess and closes the connection. Always call this when done —
        an unclosed MCP server keeps its child process running."""
        await self._stack.aclose()
        await asyncio.sleep(0.15)


def _make_mcp_tool(session: ClientSession, mcp_tool: Any) -> Tool:
    async def execute(tool_input: Any, ctx: ToolExecutionContext) -> Any:
        result = await session.call_tool(mcp_tool.name, arguments=tool_input or {})
        if getattr(result, "isError", False):
            text = _extract_text(result)
            raise KilnError(text or f'MCP tool "{mcp_tool.name}" returned an error.')
        structured = getattr(result, "structuredContent", None)
        return structured if structured is not None else _extract_text(result)

    return Tool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        # MCP tools declare their parameters as JSON Schema, not pydantic — this skips kiln's local
        # validation (Tool.parameters accepts a raw dict for exactly this case); the MCP server
        # validates the call on its own side when call_tool() runs.
        parameters=mcp_tool.inputSchema,
        execute=execute,
        permissions=[],
    )


async def connect_mcp(command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> MCPConnection:
    """Connects to an MCP server over stdio (spawns `command`) and exposes every tool it declares as a
    real kiln Tool — the standard way to give an agent capabilities (filesystem access, a database, a
    SaaS API, ...) that already have an MCP server, without hand-writing a Tool wrapper for each one."""
    stack = contextlib.AsyncExitStack()
    try:
        server_params = StdioServerParameters(command=command, args=args or [], env=env)
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
    except Exception as error:
        await stack.aclose()
        raise KilnError(f'Failed to connect to MCP server "{command} {" ".join(args or [])}".') from error

    tools = [_make_mcp_tool(session, mcp_tool) for mcp_tool in listed.tools]
    return MCPConnection(tools=tools, _stack=stack)
