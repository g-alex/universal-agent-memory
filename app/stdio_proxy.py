"""stdio→HTTP MCP proxy for clients without streamable-HTTP transport.

Cursor/Kiro (or any other client) launch this process as a regular stdio
MCP server; every call is transparently forwarded to the remote
universal-memory over streamable HTTP with a Bearer key.

Run:
    python -m app.stdio_proxy --url http://<vps>:8400/mcp --key <MASTER_API_KEY>

Client config:
    {"mcpServers": {"universal-memory": {
        "command": "python",
        "args": ["-m", "app.stdio_proxy",
                 "--url", "http://<vps>:8400/mcp",
                 "--key", "<KEY>"]}}}
"""
import argparse
import asyncio
import os

import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def amain(url: str, key: str) -> None:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    http_client = create_mcp_http_client(headers=headers)
    async with http_client:
        async with streamable_http_client(url, http_client=http_client) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()

                async def _list_tools(ctx, params) -> types.ListToolsResult:
                    return await session.list_tools()

                async def _call_tool(ctx, params) -> types.CallToolResult:
                    res = await session.call_tool(
                        params.name, params.arguments or {})
                    # forward everything: without structured_content a 2.x
                    # client rejects the result when the tool has an outputSchema
                    return types.CallToolResult(
                        content=list(res.content),
                        structured_content=res.structured_content,
                        is_error=res.is_error)

                server = Server(
                    "universal-memory-proxy",
                    on_list_tools=_list_tools,
                    on_call_tool=_call_tool,
                )
                async with stdio_server() as (stdin, stdout):
                    await server.run(
                        stdin, stdout,
                        server.create_initialization_options(),
                    )


def main() -> None:
    p = argparse.ArgumentParser(description="stdio→HTTP MCP proxy")
    p.add_argument("--url", default=os.getenv(
        "MEMORY_MCP_URL", "http://localhost:8400/mcp"))
    p.add_argument("--key", default=os.getenv("MEMORY_MCP_KEY", ""))
    a = p.parse_args()
    asyncio.run(amain(a.url, a.key))


if __name__ == "__main__":
    main()
