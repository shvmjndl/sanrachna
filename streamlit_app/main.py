"""Sanrachna Streamlit test UI."""

from __future__ import annotations

import json
import os
import time

import requests
import streamlit as st

st.set_page_config(page_title="Sanrachna", page_icon="📜", layout="wide")

DEFAULT_API_URL = os.environ.get("SANRACHNA_API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
st.title("📜 Sanrachna v0.3")
st.caption("Indic-first document intelligence — hybrid search, knowledge graph, agent tools")

tab_ingest, tab_search, tab_agent, tab_docs, tab_graph, tab_costs = st.tabs(
    ["Ingest", "Search", "Agent", "Documents", "Graph", "Costs"]
)

with tab_ingest:
    st.header("Upload Document")
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Language (fallback)",
            ["en-IN", "hi-IN", "ta-IN", "te-IN", "bn-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN"],
        )
    with col2:
        auto_lid = st.checkbox("Auto-detect language (requires Sarvam)", value=False)
        enable_entities = st.checkbox("Entity extraction", value=True)
        enable_graph = st.checkbox("Knowledge graph", value=True)
        enable_captions = st.checkbox("Image captions", value=True)

    uploaded = st.file_uploader(
        "Upload PDF, image, ZIP, DOCX, PPTX, or EML",
        type=["pdf", "png", "jpg", "jpeg", "zip", "txt", "docx", "pptx", "eml"],
    )

    if uploaded and st.button("Process Document", type="primary"):
        with st.spinner("Uploading..."):
            resp = requests.post(
                f"{API_URL}/v1/ingest",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={
                    "language": language,
                    "auto_detect_language": str(auto_lid).lower(),
                    "enable_entities": str(enable_entities).lower(),
                    "enable_graph": str(enable_graph).lower(),
                    "enable_captions": str(enable_captions).lower(),
                },
                timeout=60,
            )
        if resp.status_code != 200:
            st.error(f"Upload failed: {resp.text}")
        else:
            job = resp.json()
            job_id = job["job_id"]
            st.success(f"Job queued: `{job_id}`")
            st.session_state["last_job_id"] = job_id

            progress = st.progress(0)
            status_text = st.empty()
            for _ in range(180):
                status_resp = requests.get(f"{API_URL}/v1/jobs/{job_id}", timeout=10)
                if status_resp.status_code != 200:
                    break
                status = status_resp.json()
                progress.progress(min(int(status.get("progress", 0)), 100) / 100)
                status_text.text(
                    f"Status: {status['status']} — "
                    f"{status.get('documents_processed', 0)}/{status.get('documents_total', 0)} docs"
                )
                if status["status"] in ("completed", "failed"):
                    if status["status"] == "completed":
                        st.success("Processing complete!")
                        if status.get("result"):
                            result = status["result"]
                            st.session_state["corpus_id"] = result.get("corpus_id")
                            doc_ids = result.get("document_ids") or []
                            if doc_ids:
                                st.session_state["active_document_id"] = doc_ids[0]
                                st.session_state["last_document_id"] = doc_ids[0]
                        st.json(status)
                    else:
                        st.error(f"Failed: {status.get('error')}")
                    break
                time.sleep(3)

with tab_search:
    st.header("Search — find relevant chunks")
    st.caption(
        "Retrieves matching text from your documents. "
        "Use this to explore what's indexed. Agent tab answers questions using this under the hood."
    )
    query = st.text_input("Search query", placeholder="GST invoice payment terms...")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Results", 1, 20, 5)
    with col2:
        rerank = st.checkbox("Rerank results (LLM)", value=True, help="Cohere rerank if COHERE_API_KEY is set")
    with col3:
        expand_parent = st.checkbox("Expand parent context", value=False)

    doc_filter = st.session_state.get("active_document_id", "")
    if doc_filter:
        st.caption(f"Active document: `{doc_filter}`")

    if st.button("Search") and query:
        payload = {
            "query": query,
            "top_k": top_k,
            "hybrid": True,
            "rerank": rerank,
            "expand_parent": expand_parent,
        }
        if doc_filter:
            payload["document_id"] = doc_filter
        resp = requests.post(
            f"{API_URL}/v1/search",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.write(f"Found **{data['total']}** results")
            if data.get("graph_context"):
                with st.expander("Knowledge Graph Context"):
                    st.markdown(data["graph_context"])
            for r in data["results"]:
                with st.expander(f"[{r['score']:.3f}] {r['chunk_id']} — {r['source']}"):
                    st.markdown(r["text"])
                    st.caption(
                        f"Doc: {r['document_id']} | Page: {r.get('page')} | Section: {r.get('section')}"
                    )
        else:
            st.error(resp.text)

with tab_agent:
    st.header("Agent Query (11 Tools, Multi-step)")
    st.caption(
        "search_documents · graph_lookup · entity_search · table_query · "
        "list_documents · get_document_summary · compliance_check · "
        "translate_text · compare_documents · summarize_evidence · calculate"
    )
    agent_query = st.text_area(
        "Ask a question",
        placeholder="What are the parking charges?",
    )
    max_steps = st.slider("Max agent steps", 1, 12, 8)
    with st.expander("Advanced scope (optional)"):
        agent_doc_id = st.text_input(
            "Document ID override",
            value=st.session_state.get("active_document_id", ""),
            key="agent_document",
        )
        corpus_id = st.text_input(
            "Corpus ID override",
            value="",
            key="agent_corpus",
        )

    active_doc = st.session_state.get("active_document_id")
    if active_doc:
        st.caption(f"Active document: `{active_doc}` (set automatically after ingest)")
    else:
        st.caption("No document selected — API uses the most recently ingested document.")

    if st.button("Run Agent") and agent_query:
        payload = {
            "query": agent_query,
            "top_k": 5,
            "max_steps": max_steps,
        }
        active_doc = agent_doc_id or st.session_state.get("active_document_id")
        if active_doc:
            payload["document_id"] = active_doc
        if corpus_id:
            payload["corpus_id"] = corpus_id
        with st.spinner("Agent thinking..."):
            resp = requests.post(
                f"{API_URL}/v1/query",
                json=payload,
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.markdown("### Answer")
            st.markdown(data["answer"])

            if data.get("steps"):
                st.markdown("### Agent Steps")
                for i, step in enumerate(data["steps"], 1):
                    st.code(f"Step {i}: {step['tool']}({json.dumps(step['input'])})", language="python")
                    st.caption(step["output_summary"][:200])

            if data.get("entities"):
                st.markdown("### Entities Found")
                st.dataframe(
                    [{"type": e["type"], "text": e["text"], "id": e["id"]} for e in data["entities"]],
                    width="stretch",
                )

            if data.get("tables"):
                st.markdown("### Tables Found")
                for t in data["tables"]:
                    st.markdown(f"**{t.get('caption', t.get('table_id'))}**")
                    if t.get("headers"):
                        st.dataframe(t.get("rows", []), width="stretch")

            if data.get("citations"):
                st.markdown("### Citations")
                for c in data["citations"]:
                    st.info(f"**{c['chunk_id']}** (score={c['score']:.3f}): {c['text'][:200]}...")
        else:
            st.error(resp.text)

with tab_docs:
    st.header("Documents")
    col_refresh, col_corpus = st.columns([1, 3])
    with col_refresh:
        refresh = st.button("Refresh", type="primary")
    with col_corpus:
        docs_corpus_filter = st.text_input(
            "Filter by corpus ID",
            value="",
            key="docs_corpus_filter",
            placeholder="Leave empty to list all documents",
        )

    if refresh or "documents_cache" not in st.session_state:
        params = {}
        if docs_corpus_filter:
            params["corpus_id"] = docs_corpus_filter
        resp = requests.get(f"{API_URL}/v1/documents", params=params, timeout=10)
        if resp.status_code == 200:
            st.session_state["documents_cache"] = resp.json()
        else:
            st.error(resp.text)
            st.session_state["documents_cache"] = []

    docs = st.session_state.get("documents_cache", [])
    if not docs:
        st.info("No documents found. Upload one in the Ingest tab.")
    else:
        st.caption(f"{len(docs)} document(s)")
        for doc in docs:
            doc_id = doc["document_id"]
            with st.expander(f"{doc['filename']} — `{doc_id}`", expanded=False):
                st.json(doc)
                c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
                with c1:
                    view = st.button("View details", key=f"view_{doc_id}")
                with c2:
                    use_agent = st.button("Use in Agent", key=f"use_{doc_id}")
                with c3:
                    delete = st.button("Delete", key=f"delete_{doc_id}", type="secondary")
                with c4:
                    confirm = st.checkbox(
                        "Confirm permanent delete",
                        key=f"confirm_{doc_id}",
                    )

                if use_agent:
                    st.session_state["corpus_id"] = doc.get("corpus_id", "")
                    st.session_state["active_document_id"] = doc_id
                    st.session_state["last_document_id"] = doc_id
                    st.success(f"Set active document to `{doc_id}`")

                if delete:
                    if not confirm:
                        st.warning("Check **Confirm permanent delete** first.")
                    else:
                        del_resp = requests.delete(
                            f"{API_URL}/v1/documents/{doc_id}",
                            timeout=30,
                        )
                        if del_resp.status_code == 200:
                            st.success(f"Deleted `{doc_id}`")
                            if st.session_state.get("active_document_id") == doc_id:
                                st.session_state.pop("active_document_id", None)
                            st.session_state.pop("documents_cache", None)
                            st.rerun()
                        else:
                            st.error(del_resp.text)

                if view:
                    detail = requests.get(
                        f"{API_URL}/v1/documents/{doc_id}", timeout=10
                    )
                    if detail.status_code == 200:
                        ko = detail.json()
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Chunks", len(ko.get("chunks", [])))
                        m2.metric("Entities", len(ko.get("entities", [])))
                        m3.metric("Tables", len(ko.get("tables", [])))
                        m4.metric("Relations", len(ko.get("relationships", [])))
                        st.markdown(ko.get("markdown", "")[:3000])
                        with st.expander("Full JSON"):
                            st.json(ko)
                    else:
                        st.error(detail.text)

with tab_graph:
    st.header("Corpus Knowledge Graph")
    corpus_id = st.text_input("Corpus ID", value=st.session_state.get("corpus_id", ""), key="graph_corpus")
    entity_q = st.text_input("Search entities", placeholder="Acme, GSTIN, vendor...")

    if st.button("Load Graph") and corpus_id:
        resp = requests.get(f"{API_URL}/v1/graph/{corpus_id}", timeout=10)
        if resp.status_code == 200:
            graph = resp.json()
            st.metric("Nodes", len(graph.get("nodes", [])))
            st.metric("Edges", len(graph.get("edges", [])))
            if graph.get("nodes"):
                st.dataframe(graph["nodes"], width="stretch")
            if graph.get("edges"):
                st.dataframe(graph["edges"], width="stretch")

    if st.button("Search Entities") and entity_q:
        resp = requests.get(
            f"{API_URL}/v1/graph/entities/search",
            params={"q": entity_q, "corpus_id": corpus_id or None},
            timeout=10,
        )
        if resp.status_code == 200:
            st.dataframe(resp.json(), width="stretch")

with tab_costs:
    st.header("Cost Tracking")
    resp = requests.get(f"{API_URL}/v1/costs/summary", timeout=10)
    if resp.status_code == 200:
        costs = resp.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Sarvam Cost (INR)", f"₹{costs['sarvam_cost_inr']:.2f}")
        c2.metric("LiteLLM Calls", costs["litellm_calls"])
        c3.metric("Total Tokens", f"{costs['total_input_tokens'] + costs['total_output_tokens']:,}")
        st.subheader("Recent Operations")
        st.dataframe(costs["recent"], width="stretch")
    else:
        st.warning("Cost data unavailable")

st.sidebar.markdown("---")
st.sidebar.markdown("**v0.3 Features**")
st.sidebar.markdown("- Cohere rerank-v3.5")
st.sidebar.markdown("- 11 agent tools")
st.sidebar.markdown("- Compliance validation")
st.sidebar.markdown("- Sarvam translation")
st.sidebar.markdown("- Document comparison")
st.sidebar.markdown("**LiteLLM UI:** http://localhost:4000/ui")
