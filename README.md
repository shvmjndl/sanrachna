# Sanrachna v0.3

**Indic-first document-to-knowledge platform.** Upload any document → get an agent-ready **Knowledge Object** with hybrid search, knowledge graph, grounded citations, schema extraction, cross-document reasoning, and a 13-tool agent.

Unlike parsing-only services (Unstructured, MinerU, Nanonets, LlamaParse), Sanrachna is an end-to-end **knowledge layer**: ingest, index, query, extract, compare, evaluate, and integrate — out of the box.

**What you get that parsers alone don't:**
- Hybrid retrieval (dense + BM25 + RRF + rerank) with evidence confidence
- Parent-child chunking with bbox provenance on PDFs
- Knowledge graph + multi-hop traversal across entities
- Schema-on-read extraction with built-in templates
- Cross-document compare, contradiction detection, and document families
- Pluggable digitizers (Local / Sarvam / MinerU / Vision / Tesseract) with content-aware auto-routing and fallback chains
- Agent with 13 tools, SSE streaming, MCP bridge, and optional LangChain integration

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Services & URLs](#services--urls)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Features](#features)
- [API Reference](#api-reference)
- [Python SDK & CLI](#python-sdk--cli)
- [Integrations](#integrations)
- [Streamlit UI](#streamlit-ui)
- [Agent Tools](#agent-tools)
- [Building RAG vs using the Agent](#building-rag-vs-using-the-agent)
- [Supported file formats](#supported-file-formats)
- [Project Structure](#project-structure)

---

## Architecture

```
Upload (PDF, DOCX, ZIP, image, EML, …)
        │
        ▼
┌───────────────────┐     Celery worker
│  POST /v1/ingest  │ ──► Digitize (Local | Sarvam | MinerU | Vision | Tesseract — auto-routing + fallback)
└───────────────────┘     Chunk (parent-child) + tables + entities + graph
        │                 Embed + index (Qdrant + Postgres BM25)
        ▼
   Knowledge Object (JSONB) — chunks, tables, entities, graph, bbox, quality
        │
        ├─► POST /v1/search        Hybrid retrieval + evidence confidence
        ├─► POST /v1/query         Agent (13 tools) + citations + bbox
        ├─► POST /v1/query/stream  SSE streaming agent steps
        ├─► POST /v1/extract       Schema-on-read + template_id
        ├─► POST /v1/compare       Cross-document entity/table/field diffs
        ├─► POST /v1/contradictions Conflict detection across docs
        ├─► POST /v1/graph/traverse Multi-hop graph traversal
        ├─► GET  /v1/templates     Built-in extraction schemas
        ├─► POST /v1/translate     Query/text translation (Sarvam)
        ├─► POST /v1/webhooks      Ingest completion webhooks
        ├─► GET  /v1/families      Document family clustering
        └─► POST /v1/eval          RAG/agent quality harness
```

### Search pipeline

```
Query → [optional Sarvam translate] → Dense (Qdrant) + BM25 (Postgres tsvector)
                                              ↓
                                        RRF Fusion → Rerank → Top-K
                                              ↓
                              Cohere rerank-v3.5 (optional) → LLM rerank fallback
                                              ↓
                              Evidence confidence + graph context + bbox citations
```

### Digitizer routing (`DIGITIZATION_PROVIDER=auto`)

Sanrachna probes each PDF (text density, Indic script ratio, page count) and picks the best primary backend. If the chosen backend returns empty or placeholder text, it **falls through** the full chain automatically:

**primary → MinerU → Sarvam → Vision (LiteLLM) → Tesseract → Local**

| Signal | Routed to |
|--------|-----------|
| Text-based PDF (high density) | **Local** (PyMuPDF + bbox) |
| Low text density / scanned layout | **MinerU** → Sarvam → **Vision** → Tesseract → Local |
| Images (PNG/JPG) | Sarvam → **Vision** → Tesseract → Local |
| Indic script detected | **Sarvam** (or Vision / Tesseract if no Sarvam key) |
| Long complex PDF (15+ pages, moderate density) | **MinerU** (if enabled) |
| DOCX, TXT, EML, etc. | **Local** converter |

Set `DIGITIZATION_PROVIDER=local|sarvam|mineru|vision` to force a primary backend; unusable output still triggers fallback when other backends are configured.

**Vision OCR** renders each PDF page and sends it to a VLM through LiteLLM. Default model is **`gemini-3.5-flash`** (recommended for scanned docs). Configure via `VISION_OCR_MODEL` and comma-separated `VISION_OCR_FALLBACK_MODELS`. Models must be listed in `litellm_config.yaml` (Gemini and OpenAI entries are included out of the box).

**Tesseract OCR** is a local, no-API fallback (installed in the worker Docker image). Enable with `TESSERACT_OCR_ENABLED=true`.

> **Note:** Some OpenAI project keys allow text chat but **not** vision (`missing_scope` on image requests). Use **`GEMINI_API_KEY`** for scanned PDF OCR while keeping OpenAI for embeddings and the agent.

### Scanned PDF quick setup (Gemini vision)

```bash
cp .env.example .env

OPENAI_API_KEY=sk-...              # embeddings, NER, agent (text)
GEMINI_API_KEY=...                 # vision OCR for scanned PDFs
DIGITIZATION_PROVIDER=auto
VISION_OCR_MODEL=gemini-3.5-flash
VISION_OCR_FALLBACK_MODELS=gemini-2.5-flash

docker compose up -d --build
```

An 18-page scanned PDF takes roughly 5–10 minutes (one LiteLLM call per page).

---

## Quick Start

### English-only (no Sarvam)

You need **OpenAI** for embeddings, NER, graph, and the agent. For **scanned PDFs**, also set **`GEMINI_API_KEY`** (see [Scanned PDF quick setup](#scanned-pdf-quick-setup-gemini-vision)).

```bash
cp .env.example .env
```

Set in `.env`:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=                    # required for scanned/image PDF OCR
SARVAM_API_KEY=                    # leave empty
DIGITIZATION_PROVIDER=auto         # auto-routes scanned PDFs to Gemini vision
VISION_OCR_MODEL=gemini-3.5-flash
COHERE_API_KEY=                    # optional — LLM rerank fallback works without it
ENABLE_COHERE_RERANK=false
```

```bash
docker compose up --build
```

Test with the sample invoice:

```bash
# 1. Ingest
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@samples/english_invoice.txt" \
  -F "language=en-IN"

# Response:
# {"job_id":"a1b2c3d4-...","status":"pending","message":"Document queued for processing"}

# 2. Poll until completed
curl http://localhost:8000/v1/jobs/<job_id>

# Response (when done):
# {
#   "job_id": "...",
#   "status": "completed",
#   "progress": 100.0,
#   "documents_processed": 1,
#   "result": {
#     "corpus_id": "corp_a1b2c3d4-e0f",
#     "document_ids": ["doc_f8a2b1c3d4e5"],
#     "documents": 1
#   }
# }

# 3. Ask a question (no document_id needed — auto-scopes to latest doc)
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total invoice amount including GST?"}'
```

### Full Indic stack

```bash
cp .env.example .env
# Set SARVAM_API_KEY, OPENAI_API_KEY, and optionally GEMINI_API_KEY for vision OCR fallback
DIGITIZATION_PROVIDER=auto   # smart-routes: local / Sarvam / MinerU / Vision / Tesseract
MINERU_ENABLED=false         # set true if mineru CLI is installed
AUTO_TRANSLATE_QUERIES=true  # Hindi/Tamil/etc. queries auto-translated for search

docker compose up --build
```

Try a Hindi query after ingesting an Indic document — the agent translates, searches, and answers in the original language when Sarvam is configured.

---

## Services & URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **API** | http://localhost:8000 | REST API |
| **API docs** | http://localhost:8000/docs | Swagger UI |
| **Streamlit UI** | http://localhost:8501 | Upload, search, agent, documents |
| **LiteLLM admin** | http://localhost:4000/ui | Model routing & cost |
| **Qdrant** | http://localhost:6333 | Vector store |
| **Postgres** | localhost:5432 | Metadata + BM25 |
| **Redis** | localhost:6379 | Celery broker |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Embeddings, NER, agent (text models via LiteLLM) |
| `GEMINI_API_KEY` | — | Vision OCR for scanned PDFs (`gemini-3.5-flash` via LiteLLM) |
| `SARVAM_API_KEY` | — | Indic OCR, LID, translation |
| `DIGITIZATION_PROVIDER` | `auto` | `auto` \| `local` \| `sarvam` \| `mineru` \| `vision` |
| `MINERU_ENABLED` | `false` | Enable MinerU CLI for complex/scanned PDFs |
| `MINERU_BIN` | `mineru` | MinerU executable path |
| `MINERU_BACKEND` | `pipeline` | MinerU backend mode |
| `MINERU_TIMEOUT_SEC` | `600` | MinerU subprocess timeout (seconds) |
| `VISION_OCR_ENABLED` | `true` | Vision LLM OCR via LiteLLM when local fails |
| `VISION_OCR_MODEL` | `gemini-3.5-flash` | Primary VLM for page OCR |
| `VISION_OCR_FALLBACK_MODELS` | `gemini-2.5-flash,gpt-4o-mini` | Comma-separated VLM fallbacks per page |
| `VISION_OCR_MAX_PAGES` | `20` | Max PDF pages to send to vision OCR |
| `VISION_OCR_RENDER_SCALE` | `2.0` | PDF page render scale for vision OCR |
| `VISION_OCR_TIMEOUT_SEC` | `120` | Per-page vision OCR HTTP timeout |
| `TESSERACT_OCR_ENABLED` | `true` | Local Tesseract fallback (worker image includes `tesseract-ocr`) |
| `TESSERACT_OCR_LANG` | `eng` | Tesseract language pack(s) |
| `TESSERACT_OCR_MAX_PAGES` | `50` | Max PDF pages for Tesseract OCR |
| `TESSERACT_OCR_RENDER_SCALE` | `2.0` | PDF page render scale for Tesseract |
| `AUTO_TRANSLATE_QUERIES` | `true` | Auto-translate non-English queries before search |
| `COHERE_API_KEY` | — | Optional reranking |
| `ENABLE_COHERE_RERANK` | `false` | Enable Cohere rerank-v3.5 |
| `LITELLM_CHAT_MODEL` | `gpt-4o-mini` | Agent & extraction model |
| `LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Chunk embeddings |
| `SANRACHNA_API_URL` | `http://localhost:8000` | SDK/CLI base URL |

### What works without Sarvam

| Component | Provider |
|-----------|----------|
| TXT, MD, DOCX, PPTX, EML | Local converter |
| Text-based PDF | PyMuPDF (local) + bbox capture |
| Scanned / image PDF | **Gemini vision OCR** (via LiteLLM), MinerU CLI, or **Tesseract** |
| Embeddings, NER, graph, agent | OpenAI via LiteLLM |
| Search rerank | Cohere (optional) or LLM fallback |

### What needs Sarvam

| Component | Why |
|-----------|-----|
| Indic scanned PDF / images | Best-quality Indic script OCR |
| Hindi, Tamil, Telugu, etc. | Indic script OCR + query translation |
| Language detection (LID) | Sarvam LID API |
| `translate_text` agent tool | Sarvam Translate |
| `POST /v1/translate` | Sarvam Translate |
| Multilingual query preprocessing | Sarvam LID + Translate (`AUTO_TRANSLATE_QUERIES`) |

Without Sarvam, scanned Indic documents can still be processed via Gemini vision or Tesseract, but Sarvam generally gives better results for Indic layouts.

---

## Core Concepts

### Knowledge Object

Every ingested document becomes a versioned **Knowledge Object** — not just markdown.

```json
{
  "document_id": "doc_f8a2b1c3d4e5",
  "source": {
    "filename": "english_invoice.txt",
    "language": "en-IN",
    "document_family": "tabular_records",
    "family_signals": { "numeric_ratio": 0.42, "table_rows": 3 }
  },
  "markdown": "# INVOICE #INV-2025-0042\n...",
  "chunks": [
    {
      "id": "child_a1b2c3d4",
      "text": "Total Due: Rs. 76,700",
      "page": 1,
      "section": "paragraph",
      "chunk_type": "child",
      "bbox": { "x0": 72.0, "y0": 640.0, "x1": 280.0, "y1": 655.0 },
      "confidence": 0.94,
      "parent_id": "parent_x1y2z3",
      "parent_text": "..."
    }
  ],
  "tables": [
    {
      "id": "table_001",
      "headers": ["Description", "Qty", "Rate (INR)", "Amount (INR)"],
      "rows": [["Cloud Hosting Q1", "1", "45,000", "45,000"]]
    }
  ],
  "entities": [
    { "id": "ent_001", "type": "GSTIN", "text": "27AABCU9603R1ZM", "confidence": 0.95 }
  ],
  "relationships": [],
  "knowledge_graph": { "nodes": [], "edges": [] },
  "metadata": {
    "pages": 1,
    "quality_score": 0.87,
    "processing_time_ms": 4200
  }
}
```

### Native scope resolution

You do **not** need to pass `corpus_id` or `document_id` for every query. The API resolves scope automatically:

| Situation | Behavior |
|-----------|----------|
| Valid `document_id` | Search that document |
| Valid `corpus_id` with docs | Search corpus (single doc → auto-scope) |
| Invalid / empty corpus | Ignored, falls through |
| Only 1 doc in system | Auto-scope to it |
| Multiple docs, no hint | **Latest ingested document** |
| No docs | Empty results |

### Document families (structural, not hardcoded)

At ingest, each document is classified by layout signals:

| Family | Signals |
|--------|---------|
| `tabular_records` | Dense tables, numeric rows, record chunks |
| `key_value_form` | Label:value lines, moderate numerics |
| `narrative_longform` | Long paragraphs, low numeric density |
| `layout_rich` | High bbox coverage, mixed content |
| `sparse_short` | Very little text |
| `mixed_general` | Everything else |

### Document versioning

Re-ingest the same file or explicitly supersede a document. Sanrachna tracks `version`, `content_hash`, and `supersedes_id` on each document.

| Form field | Purpose |
|------------|---------|
| `document_id` | Pin a specific ID for a new document |
| `supersedes_id` | Replace an existing document (bumps version, re-indexes) |
| `incremental` | `true` (default) — auto-detect same content hash and version up |

```bash
# Re-ingest an updated invoice, keeping the same document_id
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@invoice_v2.pdf" \
  -F "supersedes_id=doc_f8a2b1c3d4e5"
```

---

## Features

### 1. Ingest & async processing

Upload a file; a Celery worker digitizes, chunks, enriches, and indexes it.

**Input** (multipart form):

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@invoice.pdf" \
  -F "language=en-IN" \
  -F "enable_entities=true" \
  -F "enable_graph=true" \
  -F "enable_captions=true" \
  -F "auto_detect_language=false" \
  -F "document_id=" \
  -F "supersedes_id=" \
  -F "incremental=true"
```

| Form field | Default | Description |
|------------|---------|-------------|
| `language` | `en-IN` | BCP-47 locale hint |
| `enable_entities` | `true` | Extract NER entities |
| `enable_graph` | `true` | Build knowledge graph |
| `enable_captions` | `true` | Caption images/tables |
| `auto_detect_language` | `true` | Sarvam LID on ingest |
| `document_id` | — | Optional fixed document ID |
| `supersedes_id` | — | Replace existing document |
| `incremental` | `true` | Auto-version on same content hash |

**Output**:

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "pending",
  "message": "Document queued for processing"
}
```

Poll `GET /v1/jobs/{job_id}` until `status` is `completed`. The `result` includes `corpus_id` and `document_ids`.

---

### 2. Hybrid search (RAG retrieval)

Dense vector search + BM25 full-text + RRF fusion + reranking.

**Input**:

```json
{
  "query": "total GST amount",
  "top_k": 5,
  "document_id": null,
  "corpus_id": null,
  "hybrid": true,
  "rerank": true,
  "expand_parent": false
}
```

**Output**:

```json
{
  "query": "total GST amount",
  "total": 5,
  "graph_context": "Entity: GSTIN 27AABCU9603R1ZM (ORG: Acme Technology Pvt Ltd)",
  "evidence": {
    "confidence": 0.78,
    "level": "medium",
    "reasons": ["5 citation(s) retrieved", "top score 0.842"],
    "warnings": [],
    "citation_count": 5,
    "table_support": false
  },
  "results": [
    {
      "chunk_id": "child_a1b2c3d4",
      "document_id": "doc_f8a2b1c3d4e5",
      "text": "GST (18%): Rs. 11,700",
      "score": 0.842,
      "confidence": 0.94,
      "page": 1,
      "section": "paragraph",
      "bbox": {
        "x0": 72.0, "y0": 620.0, "x1": 250.0, "y1": 635.0,
        "page_width": 595.0, "page_height": 842.0
      },
      "source": "hybrid",
      "retrieval_score": 0.031,
      "rerank_score": 8.5
    }
  ]
}
```

Use `expand_parent: true` to return full parent section text instead of the matched child chunk.

---

### 3. Agent query (full RAG + tools)

Multi-step agent with prefetch RAG, **13 tools**, evidence confidence, multilingual query support, and citations.

**Input**:

```json
{
  "query": "What is the total due including GST?",
  "top_k": 5,
  "max_steps": 8
}
```

**Output**:

```json
{
  "answer": "The total due including GST is Rs. 76,700.",
  "evidence": {
    "confidence": 0.85,
    "level": "high",
    "reasons": [
      "5 citation(s) retrieved",
      "top score 0.891",
      "structured table rows available"
    ],
    "warnings": [],
    "citation_count": 5,
    "table_support": true
  },
  "scope": {
    "document_id": "doc_f8a2b1c3d4e5",
    "corpus_id": "corp_a1b2c3d4-e0f",
    "mode": "latest",
    "query_language": "en-IN",
    "translated_query": false
  },
  "citations": [
    {
      "chunk_id": "child_a1b2c3d4",
      "document_id": "doc_f8a2b1c3d4e5",
      "text": "Total Due: Rs. 76,700",
      "score": 0.891,
      "confidence": 0.94,
      "page": 1,
      "bbox": { "x0": 72.0, "y0": 640.0, "x1": 280.0, "y1": 655.0 }
    }
  ],
  "entities": [
    { "type": "ORG", "text": "Acme Technology Pvt Ltd", "confidence": 0.8 }
  ],
  "tables": [
    {
      "table_id": "table_001",
      "headers": ["Description", "Qty", "Rate (INR)", "Amount (INR)"],
      "rows": [["Cloud Hosting Q1", "1", "45,000", "45,000"]]
    }
  ],
  "steps": [
    { "tool": "search_documents", "input": {"query": "total due GST"}, "output_summary": "..." }
  ],
  "graph": null
}
```

**Streaming** (`POST /v1/query/stream`) — Server-Sent Events with step-by-step tool output:

```
data: {"event": "start", "query": "What is the total due?"}
data: {"event": "step", "tool": "search_documents", "summary": "..."}
data: {"event": "answer", "answer": "...", "evidence": {...}, "citations": [...]}
```

---

### 4. Schema-on-read extraction

Define fields generically — no hardcoded document types. Or use a **built-in template** via `template_id`.

**Built-in templates** (`GET /v1/templates`):

| `template_id` | Use case |
|---------------|----------|
| `financial_summary` | Amounts, parties, dates, reference IDs |
| `identity_compliance` | GSTIN, PAN, IFSC, registration numbers |
| `tabular_charges` | Line-item fees, base/tax/total per charge |
| `contract_terms` | Parties, effective date, payment/termination terms |

**Input** (custom fields):

```json
{
  "document_id": "doc_f8a2b1c3d4e5",
  "fields": [
    { "name": "vendor_name", "description": "organization selling goods or services", "type": "string" },
    { "name": "total_due", "description": "final payable amount including all taxes", "type": "string" },
    { "name": "gstin", "description": "GST identification number of the vendor", "type": "string" }
  ],
  "top_k": 8
}
```

**Input** (template):

```json
{
  "document_id": "doc_f8a2b1c3d4e5",
  "template_id": "financial_summary",
  "top_k": 8
}
```

**Output**:

```json
{
  "document_id": "doc_f8a2b1c3d4e5",
  "processing_ms": 1240,
  "fields": [
    {
      "name": "vendor_name",
      "value": "Acme Technology Pvt Ltd",
      "confidence": 0.92,
      "source_chunk_ids": ["child_a1b2c3d4"],
      "source_pages": [1],
      "reasoning": "Listed as Vendor in document header"
    },
    {
      "name": "total_due",
      "value": "Rs. 76,700",
      "confidence": 0.95,
      "source_chunk_ids": ["child_e5f6g7h8"],
      "source_pages": [1],
      "reasoning": "Explicit Total Due line"
    },
    {
      "name": "gstin",
      "value": "27AABCU9603R1ZM",
      "confidence": 0.97,
      "source_chunk_ids": ["child_a1b2c3d4"],
      "source_pages": [1],
      "reasoning": "Regex-validated GSTIN entity"
    }
  ]
}
```

---

### 5. Bbox provenance & PDF highlights

Every chunk from text-based PDFs carries page coordinates. Use these to highlight sources in your UI.

**Input**:

```
GET /v1/documents/doc_f8a2b1c3d4e5/highlights?chunk_ids=child_a1b2c3d4,child_e5f6g7h8
```

**Output**:

```json
{
  "document_id": "doc_f8a2b1c3d4e5",
  "file_path": "/app/data/uploads/.../invoice.pdf",
  "highlights": [
    {
      "chunk_id": "child_a1b2c3d4",
      "page": 1,
      "text": "Total Due: Rs. 76,700",
      "bbox": {
        "x0": 72.0, "y0": 640.0, "x1": 280.0, "y1": 655.0,
        "page_width": 595.0, "page_height": 842.0
      }
    }
  ]
}
```

Normalized coordinates: `bbox.normalized()` → `{x0: 0.12, y0: 0.76, x1: 0.47, y1: 0.78}` (fractions of page size).

---

### 6. Document families

Find structurally similar documents in your corpus.

**Input**:

```
GET /v1/families?document_id=doc_f8a2b1c3d4e5&min_similarity=0.55&limit=20
```

**Output**:

```json
{
  "anchor_document_id": "doc_f8a2b1c3d4e5",
  "family": "tabular_records",
  "signals": {
    "anchor": { "numeric_ratio": 0.42, "table_rows": 3, "bbox_ratio": 0.8 },
    "family_label": "tabular_records"
  },
  "members": [
    {
      "document_id": "doc_f8a2b1c3d4e5",
      "filename": "english_invoice.txt",
      "corpus_id": "corp_a1b2c3d4-e0f",
      "document_family": "tabular_records",
      "similarity": 1.0,
      "shared_signals": { "quality_score": 0.87, "table_rows": 3 }
    },
    {
      "document_id": "doc_abc123def456",
      "filename": "vendor_bill.pdf",
      "corpus_id": "corp_xyz789",
      "document_family": "tabular_records",
      "similarity": 0.72,
      "shared_signals": { "quality_score": 0.81, "table_rows": 8 }
    }
  ]
}
```

---

### 7. RAG / agent evaluation harness

Run golden test cases and measure pass rate, recall@k, and confidence.

**Input** (`eval/fixtures/invoice_cases.json`):

```json
{
  "mode": "agent",
  "top_k": 5,
  "cases": [
    {
      "query": "What is the total due including GST?",
      "expected_contains": ["76,700", "76700"],
      "expected_not_contains": ["couldn't find"]
    },
    {
      "query": "Who is the vendor?",
      "expected_contains": ["Acme"]
    }
  ]
}
```

**Output**:

```json
{
  "metrics": {
    "total": 2,
    "passed": 2,
    "pass_rate": 1.0,
    "avg_recall_at_k": 0.0,
    "avg_confidence": 0.83
  },
  "results": [
    {
      "query": "What is the total due including GST?",
      "passed": true,
      "answer": "The total due including GST is Rs. 76,700.",
      "top_chunk_ids": ["child_e5f6g7h8", "child_a1b2c3d4"],
      "missing_terms": [],
      "forbidden_terms": [],
      "recall_at_k": 0.0,
      "confidence": 0.85
    }
  ]
}
```

---

### 8. Knowledge graph

**Corpus graph**:

```
GET /v1/graph/corp_a1b2c3d4-e0f
```

```json
{
  "nodes": [
    { "id": "ent_001", "label": "Acme Technology Pvt Ltd", "type": "ORG" }
  ],
  "edges": [
    { "source": "ent_001", "target": "ent_002", "predicate": "HAS_GSTIN", "confidence": 0.9 }
  ]
}
```

**Entity search**:

```
GET /v1/graph/entities/search?q=Acme&corpus_id=corp_a1b2c3d4-e0f
```

**Multi-hop traversal**:

```json
POST /v1/graph/traverse
{ "start_entity_query": "Acme", "max_hops": 3, "corpus_id": "corp_a1b2c3d4-e0f" }
```

---

### 9. Document management

| Action | Endpoint |
|--------|----------|
| List docs | `GET /v1/documents?corpus_id=&limit=50` |
| Get full KO | `GET /v1/documents/{document_id}` |
| Delete doc | `DELETE /v1/documents/{document_id}` |
| Highlights | `GET /v1/documents/{id}/highlights?chunk_ids=...` |

**Delete response**:

```json
{
  "deleted": true,
  "document_id": "doc_f8a2b1c3d4e5",
  "removed_chunks": 12,
  "removed_entities": 4
}
```

---

### 10. Cost tracking

```
GET /v1/costs/summary
```

```json
{
  "sarvam_cost_inr": 2.50,
  "litellm_calls": 47,
  "total_input_tokens": 12500,
  "total_output_tokens": 3200,
  "recent": [
    { "operation": "chunk_embedding", "model": "text-embedding-3-small", "tokens": 800 }
  ]
}
```

---

### 11. Document comparison

Diff entities, tables, and label:value fields between two documents.

**Input**:

```json
{
  "document_id_a": "doc_invoice_v1",
  "document_id_b": "doc_invoice_v2"
}
```

**Output**:

```json
{
  "document_id_a": "doc_invoice_v1",
  "document_id_b": "doc_invoice_v2",
  "shared_entities": [
    { "key": "acme technology pvt ltd", "type": "ORG", "text_a": "Acme Technology Pvt Ltd", "text_b": "Acme Technology Pvt Ltd", "changed": false }
  ],
  "only_in_a": [],
  "only_in_b": [{ "key": "new line item", "type": "MONEY", "text": "Rs. 5,000" }],
  "table_diffs": [
    { "table_a": "table_001", "table_b": "table_001", "headers": ["Description", "Amount"], "rows_only_in_a": [], "rows_only_in_b": ["Support Fee | 5,000"], "shared_rows": 3 }
  ],
  "field_diffs": [
    { "label": "total due", "value_a": "Rs. 76,700", "value_b": "Rs. 81,700" }
  ],
  "summary": {
    "shared_entity_count": 4,
    "changed_entities": 1,
    "tables_a": 1,
    "tables_b": 1
  }
}
```

---

### 12. Contradiction detection

Find conflicting values between two documents or scan a corpus for conflicts.

**Between two documents**:

```json
{
  "document_id_a": "doc_invoice_v1",
  "document_id_b": "doc_invoice_v2"
}
```

**Corpus scan**:

```json
{ "corpus_id": "corp_a1b2c3d4-e0f" }
```

**Output**:

```json
{
  "document_id_a": "doc_invoice_v1",
  "document_id_b": "doc_invoice_v2",
  "contradiction_count": 2,
  "contradictions": [
    {
      "type": "label_value_mismatch",
      "field": "total due",
      "value_a": "Rs. 76,700",
      "value_b": "Rs. 81,700",
      "severity": "high"
    },
    {
      "type": "entity_value_mismatch",
      "field": "27aabce9603r1zm",
      "entity_type": "GSTIN",
      "value_a": "27AABCU9603R1ZM",
      "value_b": "27AABCE9603R1ZX",
      "severity": "high"
    }
  ],
  "comparison_summary": { "changed_entities": 1 }
}
```

---

### 13. Multi-hop graph traversal

BFS traversal from entities matching a query, up to `max_hops` relationship hops.

**Input**:

```json
{
  "start_entity_query": "Acme Technology",
  "max_hops": 3,
  "corpus_id": "corp_a1b2c3d4-e0f",
  "predicate_filter": null
}
```

**Output**:

```json
{
  "start_entities": [
    { "id": "ent_001", "type": "ORG", "text": "Acme Technology Pvt Ltd", "confidence": 0.9 }
  ],
  "paths": [
    [{ "entity_id": "ent_001", "predicate": "HAS_GSTIN", "entity_id_to": "ent_002" }]
  ],
  "nodes": [
    { "id": "ent_001", "label": "Acme Technology Pvt Ltd", "type": "ORG" },
    { "id": "ent_002", "label": "27AABCU9603R1ZM", "type": "GSTIN" }
  ],
  "edges": [
    { "source": "ent_001", "target": "ent_002", "predicate": "HAS_GSTIN", "confidence": 0.9 }
  ]
}
```

---

### 14. Translation

Translate text via Sarvam (also used internally for multilingual query preprocessing when `AUTO_TRANSLATE_QUERIES=true`).

**Input**:

```json
{
  "text": "कुल देय राशि क्या है?",
  "source_language": "auto",
  "target_language": "en-IN"
}
```

**Output**:

```json
{
  "original": "कुल देय राशि क्या है?",
  "translated": "What is the total payable amount?",
  "source_language": "hi-IN",
  "target_language": "en-IN"
}
```

---

### 15. Webhooks

Register a URL to receive events when ingest jobs complete. Payloads are signed with HMAC-SHA256 in the `X-Sanrachna-Signature` header.

**Register**:

```json
{
  "url": "https://example.com/sanrachna/hook",
  "events": ["ingest.completed"],
  "secret": "optional-custom-secret"
}
```

**Payload** (`ingest.completed`):

```json
{
  "event": "ingest.completed",
  "payload": {
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "corpus_id": "corp_a1b2c3d4-e0f",
    "document_ids": ["doc_f8a2b1c3d4e5"]
  }
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/v1/ingest` | Upload & queue document |
| `GET` | `/v1/jobs/{job_id}` | Ingest job status |
| `GET` | `/v1/documents` | List indexed documents |
| `GET` | `/v1/documents/{id}` | Full knowledge object |
| `DELETE` | `/v1/documents/{id}` | Delete document + indexes |
| `GET` | `/v1/documents/{id}/highlights` | Bbox highlights for chunks |
| `POST` | `/v1/search` | Hybrid search + evidence |
| `POST` | `/v1/query` | Agent with tools |
| `POST` | `/v1/query/stream` | Agent with SSE step streaming |
| `POST` | `/v1/extract` | Schema-on-read extraction (`fields` or `template_id`) |
| `POST` | `/v1/compare` | Compare two documents |
| `POST` | `/v1/contradictions` | Detect conflicting values |
| `POST` | `/v1/graph/traverse` | Multi-hop graph traversal |
| `GET` | `/v1/templates` | List extraction templates |
| `GET` | `/v1/templates/{id}` | Get template fields |
| `POST` | `/v1/translate` | Translate text (Sarvam) |
| `POST` | `/v1/webhooks` | Register webhook |
| `GET` | `/v1/webhooks` | List webhooks |
| `POST` | `/v1/eval` | RAG/agent eval harness |
| `GET` | `/v1/families` | Document family clustering |
| `GET` | `/v1/graph/{corpus_id}` | Corpus knowledge graph |
| `GET` | `/v1/graph/entities/search` | Entity search |
| `GET` | `/v1/costs/summary` | Cost tracking |

---

## Python SDK & CLI

### SDK (`client.py`)

```python
from pathlib import Path
from client import SanrachnaClient

client = SanrachnaClient("http://localhost:8000")

# Ingest and wait
job = client.ingest(Path("samples/english_invoice.txt"))
status = client.wait_for_job(job["job_id"])
doc_id = status["result"]["document_ids"][0]

# Agent (auto-scopes to latest doc if no ID given)
response = client.query("What is the total due including GST?")
print(response["answer"])
print(response["evidence"])   # confidence, level, warnings
print(response["citations"])  # chunk_id, bbox, score

# RAG retrieval only
hits = client.search("GST amount", top_k=5)
context = client.build_rag_context(hits)  # LLM-ready string

# Schema extraction
fields = client.extract(
    [
        {"name": "total_due", "description": "final payable amount including tax"},
        {"name": "vendor_name", "description": "seller organization"},
    ],
    document_id=doc_id,
)

# PDF highlights
highlights = client.highlights(doc_id, ["child_a1b2c3d4"])

# Document families
family = client.families(document_id=doc_id)

# Eval
results = client.eval_cases([
    {"query": "What is total due?", "expected_contains": ["76,700"]},
])

# Compare & contradictions
diff = client.compare("doc_a", "doc_b")
conflicts = client.contradictions(document_id_a="doc_a", document_id_b="doc_b")

# Graph traversal & templates
graph = client.graph_traverse("Acme Corp", max_hops=3, corpus_id="corp_xxx")
templates = client.list_templates()
fields = client.extract_template("financial_summary", document_id=doc_id)

# Streaming agent
for event in client.query_stream("What is the total due?"):
    print(event)

# Webhooks & incremental re-ingest
client.register_webhook("https://example.com/hook")
client.ingest(Path("updated.pdf"), supersedes_id=doc_id)
```

### CLI (`main.py`)

```bash
# Ingest
python main.py ingest --file samples/english_invoice.txt

# Hybrid search
python main.py search --query "payment terms"

# RAG context (retrieval only — you own synthesis)
python main.py rag --file samples/english_invoice.txt --query "What is the total GST?"

# Agent (full pipeline)
python main.py agent --query "What is the total due including GST?"

# Schema extraction
python main.py extract \
  --fields-json '[{"name":"total_due","description":"final amount including GST"}]' \
  --document-id doc_xxx

# Document families
python main.py families --document-id doc_xxx

# Bbox highlights
python main.py highlights --document-id doc_xxx --chunk-ids child_abc,child_def

# Eval harness
python main.py eval --cases-file eval/fixtures/invoice_cases.json

# Compare two documents
python main.py compare --document-id-a doc_aaa --document-id-b doc_bbb

# Contradiction detection
python main.py contradictions --document-id-a doc_aaa --document-id-b doc_bbb
python main.py contradictions --corpus-id corp_xxx

# Graph traversal
python main.py traverse --entity "Acme Corp" --max-hops 3 --corpus-id corp_xxx

# Schema templates
python main.py templates
python main.py extract-template --template-id financial_summary --document-id doc_xxx

# Streaming agent (SSE)
python main.py stream --query "What is the total due?"

# Webhooks
python main.py webhook --url https://example.com/hook
python main.py webhook --list

# List documents
python main.py list
```

Set `SANRACHNA_API_URL` if the API is not on `http://localhost:8000`.

---

## Integrations

### MCP server (`mcp/server.py`)

Stdio JSON-RPC bridge exposing search, query, extract, compare, and graph tools. Useful for wiring Sanrachna into MCP-compatible agents.

```bash
SANRACHNA_API_URL=http://localhost:8000 python -m mcp.server
```

```json
{"method": "tools/list"}
{"method": "tools/call", "params": {"name": "sanrachna_search", "arguments": {"query": "GST amount"}}}
{"method": "tools/call", "params": {"name": "sanrachna_compare", "arguments": {"document_id_a": "doc_a", "document_id_b": "doc_b"}}}
{"method": "health"}
```

Available tools: `sanrachna_search`, `sanrachna_query`, `sanrachna_extract`, `sanrachna_compare`, `sanrachna_graph_traverse`.

### LangChain integration (optional)

```bash
pip install -e ".[langchain]"
```

```python
from client import SanrachnaClient
from integrations.langchain.sanrachna import SanrachnaRetriever, build_sanrachna_tools

client = SanrachnaClient()
retriever = SanrachnaRetriever(client, top_k=5)
docs = retriever.invoke("What is the total GST?")

tools = build_sanrachna_tools(client)  # sanrachna_search, sanrachna_ask, sanrachna_extract
```

---

## Streamlit UI

Open http://localhost:8501

| Tab | What it does |
|-----|--------------|
| **Ingest** | Upload PDF/image/ZIP/DOCX, poll job, auto-set active document |
| **Search** | Hybrid search with rerank toggle |
| **Agent** | Multi-step agent — no IDs needed after ingest |
| **Documents** | List, view, delete, "Use in Agent" |
| **Graph** | Corpus knowledge graph explorer |
| **Costs** | Sarvam + LiteLLM cost summary |

After ingest completes, the **Agent** and **Search** tabs automatically use the ingested document. Override scope in the "Advanced scope" expander if needed.

---

## Agent Tools

The agent has 13 tools and prefetches RAG + table context before the tool loop.

| Tool | Purpose |
|------|---------|
| `search_documents` | Hybrid search with rerank |
| `graph_lookup` | Entity + relationship traversal |
| `graph_traverse` | Multi-hop BFS graph traversal |
| `entity_search` | Cross-doc canonical entity lookup |
| `table_query` | Structured table search |
| `list_documents` | Corpus document listing |
| `get_document_summary` | Doc metadata + excerpt |
| `compliance_check` | GSTIN (checksum)/PAN/IFSC/Aadhaar validation |
| `translate_text` | Sarvam cross-lingual translation |
| `compare_documents` | Full entity/table/field diff between docs |
| `detect_contradictions` | Conflicting values between two documents |
| `summarize_evidence` | Brief from gathered citations |
| `calculate` | Arithmetic (GST math, totals, percentages) |

---

## Building RAG vs using the Agent

| Approach | When to use | How |
|----------|-------------|-----|
| **RAG (DIY)** | You control the LLM, prompts, UI | `POST /v1/search` → `build_rag_context()` → your chat completion |
| **Agent (turnkey)** | You want answers + citations + tools | `POST /v1/query` or `POST /v1/query/stream` |
| **Extract (structured)** | You need JSON fields, not prose | `POST /v1/extract` with `fields` or `template_id` |
| **Compare / audit** | Diff two document versions | `POST /v1/compare` + `POST /v1/contradictions` |
| **Graph reasoning** | Entity-centric multi-hop queries | `POST /v1/graph/traverse` |

```python
# RAG pattern
hits = client.search("total GST", top_k=5)
context = client.build_rag_context(hits)
# → pass context to your LLM

# Agent pattern
answer = client.query("What is the total GST?")
# → answer + citations + evidence + bbox

# Streaming agent
for event in client.query_stream("What is the total GST?"):
    if event.get("event") == "answer":
        print(event["answer"])

# Structured extraction via template
data = client.extract_template("financial_summary", document_id=doc_id)

# Cross-document audit
diff = client.compare("doc_v1", "doc_v2")
conflicts = client.contradictions(document_id_a="doc_v1", document_id_b="doc_v2")
```

---

## Supported file formats

| Format | Extensions | Digitizer (auto mode) |
|--------|------------|----------------------|
| PDF | `.pdf` | Local (text) → MinerU → Sarvam → **Gemini vision** → Tesseract |
| Images | `.png`, `.jpg`, `.jpeg` | Sarvam → **Gemini vision** → Tesseract |
| Office | `.docx`, `.pptx` | Local converter |
| Email | `.eml` | Local converter |
| Text | `.txt`, `.md`, `.csv` | Direct |
| Archive | `.zip` | Extracted + processed per file |

**MinerU setup** (optional, for complex/scanned PDFs):

```bash
pip install mineru   # or install per MinerU docs
# In .env:
MINERU_ENABLED=true
MINERU_BIN=mineru
```

**Gemini vision OCR** (recommended for scanned PDFs without Sarvam):

```bash
# In .env:
GEMINI_API_KEY=...
VISION_OCR_MODEL=gemini-3.5-flash
VISION_OCR_FALLBACK_MODELS=gemini-2.5-flash
```

Models are pre-configured in `litellm_config.yaml`. LiteLLM must receive `GEMINI_API_KEY` (set automatically via `docker compose`).

**Tesseract** runs inside the worker container with no API key. Disable with `TESSERACT_OCR_ENABLED=false` if you only want cloud OCR.

---

## Project Structure

```
sanrachna/
├── app/                  # FastAPI app, routes, schemas, DB models
│   └── api/routes/       # ingest, search, query, extract, compare, graph, …
├── pipeline/             # Digitize, chunk, enrich, index, extract
│   ├── digitizer_router.py  # Content-aware routing (local/sarvam/mineru/vision/tesseract)
│   ├── digitizer_utils.py   # Fallback chain + output quality checks
│   ├── vision_ocr.py        # VLM page OCR via LiteLLM (Gemini, OpenAI, …)
│   ├── tesseract_ocr.py     # Local Tesseract fallback
│   ├── mineru.py            # MinerU CLI backend
│   └── table_merge.py       # Cross-page table merge
├── services/             # Agent, graph, confidence, compare, webhooks, …
├── worker/               # Celery ingest tasks + versioning + webhooks
├── integrations/         # LangChain retriever + tools (optional)
├── mcp/                  # Stdio MCP tool bridge
├── streamlit_app/        # Test UI
├── eval/                 # Eval harness + fixtures
├── client.py             # Python SDK
├── main.py               # CLI demo
├── samples/              # Test documents
├── .env.example          # All configuration variables
└── docker-compose.yml    # Full stack (api, worker, postgres, redis, qdrant, litellm)
```

---

## Re-ingest & upgrades

- **Incremental re-ingest**: pass `supersedes_id` or upload the same file with `incremental=true` (default) to bump `version` and refresh indexes.
- **New capabilities on old docs**: bbox provenance, quality scores, document family metadata, Gemini vision OCR, Tesseract fallback, and digitizer routing improvements are computed at ingest time — re-ingest to pick them up.
- **Webhook notifications**: register `POST /v1/webhooks` to get `ingest.completed` callbacks instead of polling `GET /v1/jobs/{job_id}`.

---

## License

Private / internal — configure as needed.
