"""The synthetic screening benchmark: one agent per candidate material.

`benchmark` holds its knobs; `physics`, `prompts`, `sim` and `verify` its seeded workload;
`mcp_server` the reference-data MCP server; `stub` a model for runs without GPUs; `tools` the
simulation tool's arguments and budget; and `agent` the agent definition the AUTONMI framework runs.
Only `agent`, `sim`, `stub`, `tools` and `mcp_server` need the framework's packages.
"""
