#!/usr/bin/env python3
"""
Sanrachna client demo — ingest documents, build RAG, or run the agent.

Prerequisites:
  docker compose up --build
  export OPENAI_API_KEY=...   # required for embeddings / agent

Examples:
  # Ingest a sample invoice, then run hybrid search (RAG retrieval)
  python main.py rag --file samples/english_invoice.txt --query "What is the total GST?"

  # Same flow, but let Sanrachna's agent synthesize the answer
  python main.py agent --file samples/english_invoice.txt --query "What is the total due?"

  # Query documents already in the index (no upload)
  python main.py agent --query "What are the parking charges?"

  # Search only
  python main.py search --query "payment terms"

Environment:
  SANRACHNA_API_URL   default http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from client import SanrachnaClient, SanrachnaError, DEFAULT_API_URL


def ingest_file(
    client: SanrachnaClient,
    file_path: Path,
    *,
    language: str = "en-IN",
) -> tuple[str | None, str | None]:
    """Upload a file and block until indexing finishes. Returns (document_id, corpus_id)."""
    print(f"Ingesting {file_path} ...")
    queued = client.ingest(file_path, language=language)
    job_id = queued["job_id"]
    print(f"Queued job: {job_id}")

    result = client.wait_for_job(
        job_id,
        on_progress=lambda status: print(
            f"  job {job_id}: {status.get('status')} "
            f"({status.get('progress', 0):.0f}%) "
            f"{status.get('documents_processed', 0)}/{status.get('documents_total', 0)}"
        ),
    )
    summary = result.get("result") or {}
    document_ids = summary.get("document_ids") or []
    corpus_id = summary.get("corpus_id")
    document_id = document_ids[0] if document_ids else None

    print(f"Done. corpus_id={corpus_id} document_id={document_id}")
    return document_id, corpus_id


def build_rag_context(search_response: dict, *, max_chars: int = 6000) -> str:
    client = SanrachnaClient()
    return client.build_rag_context(search_response, max_chars=max_chars)


def print_search_results(search_response: dict) -> None:
    print(f"\nQuery: {search_response.get('query')}")
    print(f"Hits: {search_response.get('total', 0)}")
    if search_response.get("evidence"):
        ev = search_response["evidence"]
        print(f"Evidence confidence: {ev.get('confidence')} ({ev.get('level')})")
    if search_response.get("graph_context"):
        print("\n--- Graph context ---")
        print(search_response["graph_context"])

    for hit in search_response.get("results", []):
        bbox = hit.get("bbox")
        bbox_note = f" bbox={bbox}" if bbox else ""
        print(
            f"\n--- {hit.get('chunk_id')} "
            f"(score={hit.get('score', 0):.3f}, conf={hit.get('confidence', 1):.2f}{bbox_note}) ---"
        )
        print(hit.get("text", ""))


def print_agent_response(agent_response: dict) -> None:
    print("\n=== Answer ===")
    print(agent_response.get("answer", ""))
    if agent_response.get("evidence"):
        ev = agent_response["evidence"]
        print(f"\n=== Evidence ({ev.get('level')}, {ev.get('confidence')}) ===")
        for reason in ev.get("reasons", []):
            print(f"- {reason}")
        for warning in ev.get("warnings", []):
            print(f"! {warning}")
    if agent_response.get("scope"):
        print(f"\nScope: {agent_response['scope']}")

    steps = agent_response.get("steps") or []
    if steps:
        print("\n=== Agent steps ===")
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step.get('tool')}: {step.get('output_summary', '')[:200]}")

    citations = agent_response.get("citations") or []
    if citations:
        print("\n=== Citations ===")
        for c in citations:
            bbox = c.get("bbox")
            bbox_note = f" bbox={bbox}" if bbox else ""
            print(
                f"- {c.get('chunk_id')} ({c.get('score', 0):.3f}, "
                f"conf={c.get('confidence', 1):.2f}{bbox_note}): {c.get('text', '')[:160]}..."
            )


def run_rag(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)

    document_id = args.document_id
    if args.file:
        document_id, _ = ingest_file(client, Path(args.file), language=args.language)

    search_response = client.search(
        args.query,
        top_k=args.top_k,
        document_id=document_id,
        corpus_id=args.corpus_id,
        rerank=not args.no_rerank,
        expand_parent=args.expand_parent,
    )
    print_search_results(search_response)

    context = build_rag_context(search_response)
    print("\n=== RAG context (pass this to your LLM) ===")
    print(context[: args.context_chars])
    if len(context) > args.context_chars:
        print(f"\n... truncated ({len(context)} chars total)")

    print(
        "\nTip: wire `context` into your chat completion, e.g.\n"
        "  messages = [\n"
        "    {'role': 'system', 'content': 'Answer using only the context.'},\n"
        "    {'role': 'user', 'content': f'Context:\\n{context}\\n\\nQuestion: {args.query}'},\n"
        "  ]"
    )


def run_agent(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)

    document_id = args.document_id
    if args.file:
        document_id, _ = ingest_file(client, Path(args.file), language=args.language)

    response = client.query(
        args.query,
        top_k=args.top_k,
        max_steps=args.max_steps,
        document_id=document_id,
        corpus_id=args.corpus_id,
    )
    print_agent_response(response)


def run_search(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.search(
        args.query,
        top_k=args.top_k,
        document_id=args.document_id,
        corpus_id=args.corpus_id,
        rerank=not args.no_rerank,
        expand_parent=args.expand_parent,
    )
    print_search_results(response)


def run_ingest(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    document_id, corpus_id = ingest_file(client, Path(args.file), language=args.language)
    if document_id:
        ko = client.get_document(document_id)
        print(
            f"Indexed {ko['source']['filename']}: "
            f"{len(ko.get('chunks', []))} chunks, "
            f"{len(ko.get('entities', []))} entities, "
            f"{len(ko.get('tables', []))} tables"
        )
    print(f"Use document_id={document_id} or corpus_id={corpus_id} in later queries (optional).")


def run_list(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    docs = client.list_documents(corpus_id=args.corpus_id, limit=args.limit)
    if not docs:
        print("No documents indexed yet.")
        return
    for doc in docs:
        print(
            f"{doc['document_id']}  {doc['filename']}  "
            f"corpus={doc['corpus_id']}  status={doc['status']}"
        )


def run_extract(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    document_id = args.document_id
    if args.file:
        document_id, _ = ingest_file(client, Path(args.file), language=args.language)

    fields = json.loads(args.fields_json)
    response = client.extract(
        fields,
        document_id=document_id,
        corpus_id=args.corpus_id,
        top_k=args.top_k,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_families(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.families(
        document_id=args.document_id,
        corpus_id=args.corpus_id,
        min_similarity=args.min_similarity,
        limit=args.limit,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_eval(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    cases = json.loads(Path(args.cases_file).read_text())
    response = client.eval_cases(cases, mode=args.mode, top_k=args.top_k)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_highlights(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    chunk_ids = [c.strip() for c in args.chunk_ids.split(",") if c.strip()]
    response = client.highlights(args.document_id, chunk_ids)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_compare(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.compare(args.document_id_a, args.document_id_b)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_contradictions(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.contradictions(
        document_id_a=args.document_id_a,
        document_id_b=args.document_id_b,
        corpus_id=args.corpus_id,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_graph_traverse(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.graph_traverse(
        args.entity,
        max_hops=args.max_hops,
        corpus_id=args.corpus_id,
        document_id=args.document_id,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_templates(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    if args.template_id:
        response = client._request("GET", f"/v1/templates/{args.template_id}")
    else:
        response = client.list_templates()
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_extract_template(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    document_id = args.document_id
    if args.file:
        document_id, _ = ingest_file(client, Path(args.file), language=args.language)
    response = client.extract_template(
        args.template_id,
        document_id=document_id,
        corpus_id=args.corpus_id,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_translate(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    response = client.translate(
        args.text,
        source_language=args.source_language,
        target_language=args.target_language,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_webhook(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)
    if args.list:
        response = client.list_webhooks()
    else:
        events = [e.strip() for e in args.events.split(",") if e.strip()]
        response = client.register_webhook(args.url, events=events)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def run_stream(args: argparse.Namespace) -> None:
    client = SanrachnaClient(args.api_url)

    document_id = args.document_id
    if args.file:
        document_id, _ = ingest_file(client, Path(args.file), language=args.language)

    for event in client.query_stream(
        args.query,
        top_k=args.top_k,
        max_steps=args.max_steps,
        document_id=document_id,
        corpus_id=args.corpus_id,
    ):
        kind = event.get("event", "message")
        if kind == "step":
            print(f"[step] {event.get('tool')}: {event.get('summary', '')[:200]}")
        elif kind == "answer":
            print("\n=== Answer ===")
            print(event.get("answer", ""))
            if event.get("evidence"):
                ev = event["evidence"]
                print(f"\nEvidence: {ev.get('level')} ({ev.get('confidence')})")
        else:
            print(json.dumps(event, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanrachna demo client — RAG retrieval and agent queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Sanrachna API base URL (default: {DEFAULT_API_URL})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Upload and index a document")
    ingest_p.add_argument("--file", required=True, type=Path, help="Path to PDF, TXT, DOCX, etc.")
    ingest_p.add_argument("--language", default="en-IN")
    ingest_p.set_defaults(func=run_ingest)

    search_p = sub.add_parser("search", help="Hybrid search only (retrieval)")
    search_p.add_argument("--query", required=True)
    search_p.add_argument("--file", type=Path, help="Ingest this file before searching")
    search_p.add_argument("--document-id")
    search_p.add_argument("--corpus-id")
    search_p.add_argument("--top-k", type=int, default=5)
    search_p.add_argument("--no-rerank", action="store_true")
    search_p.add_argument("--expand-parent", action="store_true")
    search_p.add_argument("--language", default="en-IN")
    search_p.set_defaults(func=run_search)

    rag_p = sub.add_parser("rag", help="Retrieve chunks and build an LLM-ready context block")
    rag_p.add_argument("--query", required=True)
    rag_p.add_argument("--file", type=Path, help="Ingest this file before searching")
    rag_p.add_argument("--document-id")
    rag_p.add_argument("--corpus-id")
    rag_p.add_argument("--top-k", type=int, default=5)
    rag_p.add_argument("--no-rerank", action="store_true")
    rag_p.add_argument("--expand-parent", action="store_true")
    rag_p.add_argument("--context-chars", type=int, default=4000)
    rag_p.add_argument("--language", default="en-IN")
    rag_p.set_defaults(func=run_rag)

    agent_p = sub.add_parser("agent", help="Ask the built-in agent (search + tools + synthesis)")
    agent_p.add_argument("--query", required=True)
    agent_p.add_argument("--file", type=Path, help="Ingest this file before querying")
    agent_p.add_argument("--document-id")
    agent_p.add_argument("--corpus-id")
    agent_p.add_argument("--top-k", type=int, default=5)
    agent_p.add_argument("--max-steps", type=int, default=8)
    agent_p.add_argument("--language", default="en-IN")
    agent_p.set_defaults(func=run_agent)

    list_p = sub.add_parser("list", help="List indexed documents")
    list_p.add_argument("--corpus-id")
    list_p.add_argument("--limit", type=int, default=50)
    list_p.set_defaults(func=run_list)

    extract_p = sub.add_parser("extract", help="Schema-on-read field extraction")
    extract_p.add_argument(
        "--fields-json",
        required=True,
        help='JSON list, e.g. \'[{"name":"total_due","description":"final payable amount"}]\'',
    )
    extract_p.add_argument("--file", type=Path)
    extract_p.add_argument("--document-id")
    extract_p.add_argument("--corpus-id")
    extract_p.add_argument("--top-k", type=int, default=8)
    extract_p.add_argument("--language", default="en-IN")
    extract_p.set_defaults(func=run_extract)

    families_p = sub.add_parser("families", help="Find structurally similar document family")
    families_p.add_argument("--document-id")
    families_p.add_argument("--corpus-id")
    families_p.add_argument("--min-similarity", type=float, default=0.55)
    families_p.add_argument("--limit", type=int, default=20)
    families_p.set_defaults(func=run_families)

    eval_p = sub.add_parser("eval", help="Run RAG/agent eval cases from JSON file")
    eval_p.add_argument("--cases-file", required=True, type=Path)
    eval_p.add_argument("--mode", choices=["agent", "search"], default="agent")
    eval_p.add_argument("--top-k", type=int, default=5)
    eval_p.set_defaults(func=run_eval)

    highlights_p = sub.add_parser("highlights", help="Get bbox highlights for chunk IDs")
    highlights_p.add_argument("--document-id", required=True)
    highlights_p.add_argument("--chunk-ids", required=True, help="Comma-separated chunk IDs")
    highlights_p.set_defaults(func=run_highlights)

    compare_p = sub.add_parser("compare", help="Compare two documents (entities, tables, fields)")
    compare_p.add_argument("--document-id-a", required=True)
    compare_p.add_argument("--document-id-b", required=True)
    compare_p.set_defaults(func=run_compare)

    contradictions_p = sub.add_parser("contradictions", help="Detect contradictions between docs or in corpus")
    contradictions_p.add_argument("--document-id-a")
    contradictions_p.add_argument("--document-id-b")
    contradictions_p.add_argument("--corpus-id")
    contradictions_p.set_defaults(func=run_contradictions)

    traverse_p = sub.add_parser("traverse", help="Multi-hop knowledge graph traversal")
    traverse_p.add_argument("--entity", required=True, help="Starting entity query")
    traverse_p.add_argument("--max-hops", type=int, default=3)
    traverse_p.add_argument("--document-id")
    traverse_p.add_argument("--corpus-id")
    traverse_p.set_defaults(func=run_graph_traverse)

    templates_p = sub.add_parser("templates", help="List or view schema extraction templates")
    templates_p.add_argument("--template-id", help="Show one template by ID")
    templates_p.set_defaults(func=run_templates)

    tpl_extract_p = sub.add_parser("extract-template", help="Extract using a built-in schema template")
    tpl_extract_p.add_argument("--template-id", required=True)
    tpl_extract_p.add_argument("--file", type=Path)
    tpl_extract_p.add_argument("--document-id")
    tpl_extract_p.add_argument("--corpus-id")
    tpl_extract_p.add_argument("--language", default="en-IN")
    tpl_extract_p.set_defaults(func=run_extract_template)

    translate_p = sub.add_parser("translate", help="Translate text (requires SARVAM_API_KEY)")
    translate_p.add_argument("--text", required=True)
    translate_p.add_argument("--source-language", default="auto")
    translate_p.add_argument("--target-language", default="en-IN")
    translate_p.set_defaults(func=run_translate)

    webhook_p = sub.add_parser("webhook", help="Register or list ingest webhooks")
    webhook_p.add_argument("--url", help="Webhook URL to register")
    webhook_p.add_argument("--events", default="ingest.completed", help="Comma-separated events")
    webhook_p.add_argument("--list", action="store_true", help="List registered webhooks")
    webhook_p.set_defaults(func=run_webhook)

    stream_p = sub.add_parser("stream", help="Agent query with SSE streaming steps")
    stream_p.add_argument("--query", required=True)
    stream_p.add_argument("--file", type=Path)
    stream_p.add_argument("--document-id")
    stream_p.add_argument("--corpus-id")
    stream_p.add_argument("--top-k", type=int, default=5)
    stream_p.add_argument("--max-steps", type=int, default=8)
    stream_p.add_argument("--language", default="en-IN")
    stream_p.set_defaults(func=run_stream)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = SanrachnaClient(args.api_url)
        health = client.health()
        print(f"API ok: {health.get('service', 'sanrachna')} @ {args.api_url}")
    except SanrachnaError as exc:
        print(f"Cannot reach Sanrachna API at {args.api_url}: {exc}", file=sys.stderr)
        print("Start the stack with: docker compose up --build", file=sys.stderr)
        return 1

    try:
        args.func(args)
    except (SanrachnaError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
