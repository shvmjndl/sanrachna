"""MCP-style tool bridge for Sanrachna — exposes core API operations."""

from __future__ import annotations

import json
import os
import sys

from client import SanrachnaClient, DEFAULT_API_URL

TOOLS = [
    {
        "name": "sanrachna_search",
        "description": "Hybrid search over indexed documents",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]},
    },
    {
        "name": "sanrachna_query",
        "description": "Agent Q&A with citations and evidence",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "sanrachna_extract",
        "description": "Schema-on-read field extraction",
        "input_schema": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string"},
                "document_id": {"type": "string"},
            },
            "required": ["template_id"],
        },
    },
    {
        "name": "sanrachna_compare",
        "description": "Compare two documents for entity and field diffs",
        "input_schema": {
            "type": "object",
            "properties": {"document_id_a": {"type": "string"}, "document_id_b": {"type": "string"}},
            "required": ["document_id_a", "document_id_b"],
        },
    },
    {
        "name": "sanrachna_graph_traverse",
        "description": "Multi-hop knowledge graph traversal",
        "input_schema": {
            "type": "object",
            "properties": {"start_entity_query": {"type": "string"}, "max_hops": {"type": "integer"}},
            "required": ["start_entity_query"],
        },
    },
]


def execute_tool(client: SanrachnaClient, name: str, arguments: dict) -> dict:
    if name == "sanrachna_search":
        return client.search(arguments["query"], top_k=arguments.get("top_k", 5))
    if name == "sanrachna_query":
        return client.query(arguments["query"])
    if name == "sanrachna_extract":
        fields = client._request("GET", f"/v1/templates/{arguments['template_id']}")
        return client.extract(
            fields.get("fields", []),
            document_id=arguments.get("document_id"),
        )
    if name == "sanrachna_compare":
        return client.compare(arguments["document_id_a"], arguments["document_id_b"])
    if name == "sanrachna_graph_traverse":
        return client.graph_traverse(
            arguments["start_entity_query"],
            max_hops=arguments.get("max_hops", 3),
        )
    return {"error": f"Unknown tool: {name}"}


def main() -> None:
    """Simple stdio JSON-RPC loop for MCP-style integrations."""
    api_url = os.environ.get("SANRACHNA_API_URL", DEFAULT_API_URL)
    client = SanrachnaClient(api_url)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid json"}))
            continue

        method = req.get("method")
        if method == "tools/list":
            print(json.dumps({"tools": TOOLS}))
        elif method == "tools/call":
            params = req.get("params", {})
            result = execute_tool(client, params.get("name"), params.get("arguments", {}))
            print(json.dumps({"result": result}))
        elif method == "health":
            print(json.dumps(client.health()))
        else:
            print(json.dumps({"error": f"unknown method: {method}"}))


if __name__ == "__main__":
    main()
