"""The synthetic reference-data MCP server: one tool, `lookup_reference`, over Streamable HTTP.

    python -m screening.mcp_server --port 21234 --bench '<Benchmark as JSON>'

`start_servers(n, bench, log_dir)` starts `n` such processes on ports of their own, for the tool
services' pools to spread their clients over. On one core of an AMD EPYC 9745 (Zen 5c),
a server process serves about 3,000 lookups per second (0.25 ms of CPU each), and the tool service's
MCP clients spend about three times as much per lookup; a 100K-agent run asks for 100,000 at its start.

Stateless HTTP with JSON responses, so any client in a pool can send any call. Served by uvicorn
with a 600 s keep-alive: at uvicorn's 5 s default the server can close an idle connection just as
a client reuses it. Each call waits a latency drawn from the material, 5-50 ms by default,
as a database or web API would, and returns the reference data of `physics.lookup`.

The servers run as `python -m screening.mcp_server`, with this package's directory on their
`PYTHONPATH`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from autonmi.toolservice import free_ports, stop_servers, wait_port
from mcp.server.mcpserver import MCPServer

from . import physics
from .benchmark import Benchmark


def make_server(bench: Benchmark) -> MCPServer:
    server = MCPServer("screening-reference-data", log_level="WARNING")

    @server.tool()
    async def lookup_reference(material: str) -> dict:
        """Reference data for a material: a related compound's synthesis conditions and figure of merit."""
        t_start = time.time()
        try:
            index = physics.index_of(material.strip())
        except (ValueError, IndexError):
            raise ValueError(f"unknown material {material!r}; names look like mat-000123") from None
        if not 0 <= index < bench.agents:
            raise ValueError(f"unknown material {material!r}")
        c = physics.candidate(index, bench)
        await asyncio.sleep(physics.mcp_latency_ms(c.cid, 0, bench) / 1000)
        return {**physics.lookup(c, bench), "served_s": round(time.time() - t_start, 4)}

    return server


def _env() -> dict:
    """This process's environment, with this package's directory first on `PYTHONPATH`."""
    here = str(Path(__file__).resolve().parents[1])
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [here, os.environ.get("PYTHONPATH")])),
    }


def start_servers(
    n: int, bench: Benchmark, log_dir: str | Path, timeout: float = 60.0
) -> tuple[list[subprocess.Popen], list[str]]:
    """Starts `n` server processes on 127.0.0.1, each logging to `log_dir/mcp-server-<i>.log`, and
    waits until all listen. Returns the processes and their URLs. Stops them all if one fails."""
    log_dir = Path(log_dir)
    procs: list[subprocess.Popen] = []
    ports = free_ports(n)
    try:
        for i, port in enumerate(ports):
            with open(log_dir / f"mcp-server-{i}.log", "w") as log:
                procs.append(
                    subprocess.Popen(
                        [sys.executable, "-m", f"{__package__}.mcp_server", "--port", str(port),
                         "--bench", json.dumps(bench.asdict())],
                        stdout=log, stderr=subprocess.STDOUT, env=_env(),
                    )
                )  # fmt: skip
        for proc, port in zip(procs, ports):
            wait_port(port, timeout)
            if proc.poll() is not None:
                raise RuntimeError(f"the MCP server on port {port} exited with {proc.returncode}")
    except BaseException:
        stop_servers(procs)
        raise
    return procs, [f"http://127.0.0.1:{port}/mcp" for port in ports]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--bench", default="{}", help="the run's Benchmark, as JSON")
    p.add_argument(
        "--keep-alive", type=float, default=600.0, help="uvicorn's keep-alive timeout, seconds"
    )
    a = p.parse_args(argv)
    import uvicorn

    server = make_server(Benchmark.from_dict(json.loads(a.bench)))
    app = server.streamable_http_app(stateless_http=True, json_response=True, host=a.host)
    uvicorn.run(
        app,
        host=a.host,
        port=a.port,
        log_level="warning",
        timeout_keep_alive=a.keep_alive,
        backlog=4096,
    )


if __name__ == "__main__":
    main()
