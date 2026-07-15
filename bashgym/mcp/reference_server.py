"""Deterministic, non-destructive MCP server for Workbench contract tests.

The fixture intentionally avoids filesystem, shell, credential, and network
access.  It can run over stdio or the SDK's Streamable HTTP transport.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Annotated, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

_FIXTURES = {
    "alpha": "Alpha is the first deterministic BashGym MCP fixture.",
    "beta": "Beta is the second deterministic BashGym MCP fixture.",
}


class SumResult(BaseModel):
    total: int
    count: int
    minimum: int | None
    maximum: int | None


def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> FastMCP:
    """Build a fresh reference server with deterministic tools and capabilities."""

    server = FastMCP(
        "bashgym-mcp-reference",
        instructions="Safe deterministic fixture for MCP Workbench inspection tests.",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
        log_level="ERROR",
    )

    @server.tool()
    def read_fixture(name: Literal["alpha", "beta"] = "alpha") -> str:
        """Read one immutable public fixture by name."""

        return _FIXTURES[name]

    @server.tool(structured_output=True)
    def structured_sum(values: list[int]) -> SumResult:
        """Return deterministic summary statistics for a list of integers."""

        return SumResult(
            total=sum(values),
            count=len(values),
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
        )

    @server.tool()
    def require_positive(
        value: Annotated[int, Field(gt=0, le=1_000_000)],
    ) -> str:
        """Echo a positive integer; invalid arguments exercise schema errors."""

        return f"accepted:{value}"

    @server.tool()
    def repeat_text(
        text: Annotated[str, Field(max_length=256)],
        count: Annotated[int, Field(ge=1, le=1000)] = 1,
    ) -> str:
        """Return bounded deterministic text for result-size limit tests."""

        return text * count

    @server.tool()
    async def slow_operation(
        context: Context,
        delay_ms: Annotated[int, Field(ge=0, le=30_000)] = 1000,
        value: Annotated[str, Field(max_length=256)] = "finished",
    ) -> str:
        """Wait predictably, reporting progress; client cancellation is safe."""

        steps = 5
        for step in range(steps):
            if delay_ms:
                await asyncio.sleep(delay_ms / steps / 1000)
            await context.report_progress(
                step + 1,
                steps,
                f"reference step {step + 1}/{steps}",
            )
        return value

    @server.resource(
        "reference://guide",
        name="Reference guide",
        description="Static text describing the safe reference server.",
        mime_type="text/plain",
    )
    def reference_guide() -> str:
        return "BashGym MCP reference server: deterministic, local, and non-destructive."

    @server.resource(
        "reference://fixtures/{name}",
        name="Fixture by name",
        description="Read a named immutable fixture as a resource template.",
        mime_type="text/plain",
    )
    def fixture_resource(name: str) -> str:
        return _FIXTURES.get(name, "unknown fixture")

    @server.prompt(
        name="inspect_fixture",
        description="Create a deterministic prompt for inspecting a fixture.",
    )
    def inspect_fixture_prompt(name: Literal["alpha", "beta"] = "alpha") -> str:
        return f"Inspect the immutable {name} fixture and report its exact name."

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    build_server(host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
