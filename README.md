# CV RAG — AI Recruiter Assistant

A web application that lets recruiters ask natural-language questions about a pool of CVs and get grounded answers backed by real resume text.

You might ask: _“Who has Python and machine learning experience?”_ or _“Rank the top 3 candidates for a data engineering role.”_ The system searches your **entire CV library**, pulls the most relevant candidates, and an LLM writes a clear recommendation. Every answer can show which CV excerpts were used, so hiring decisions stay transparent.

---

## What you need before you start

This app **does not build the CV database from scratch**. It connects to an existing **Pinecone** vector index where CVs were already embedded and stored (each chunk includes the CV text, filename, and candidate name).

You will need:

| Requirement                  | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| **Pinecone account + index** | Stores embedded CV chunks                   |
| **Pinecone API key**         | Query the index                             |
| **OpenRouter API key**       | Runs the language model that writes answers |
| **Python 3.11**              | Recommended runtime (see note below)        |

---

## How it works

When someone asks a question, the app runs a **Retrieval-Augmented Generation (RAG)** pipeline:

```
User question
    │
    ▼
┌─────────────────────────────────────┐
│  1. Embed the question              │  Sentence-transformer model
│     (same family as indexing)       │  (e.g. BAAI/bge-small-en-v1.5)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. Search Pinecone                 │  Find similar CV chunks across
│     (vector similarity)             │  the whole library
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. Clean & deduplicate results     │  One entry per CV; drop empty
│                                     │  section headers; keep best chunk
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. Build context for the LLM       │  Labelled excerpts per candidate
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. Generate answer (OpenRouter)    │  HR-style analysis, grounded in CVs
└─────────────────────────────────────┘
    │
    ▼
Answer + supporting CV cards in the UI
```

### Retrieval quality

The custom retriever (`PineconeRetriever` in `rag.py`) is tuned for recruiting use cases:

- **Searches the full index**, then returns the top _N_ distinct candidates (default 5), not duplicate chunks from the same PDF.
- **Filters noise** — chunks that are only section titles (e.g. a lone `PROJECTS` header with no body) are removed or deprioritized.
- **Prefers substantive text** — when multiple chunks exist for one CV, the richest meaningful excerpt wins.
- **Ranking questions** — phrases like “rank the top 3” trigger a larger search pool so more strong candidates are considered before the shortlist is built.

### Answer style

The LLM is instructed to:

- Use **only** information from retrieved CVs (no invented skills or experience).
- Write in plain language for HR readers — no “search scores”, “rank #”, or technical retrieval jargon in the answer.
- For **ranking / top-N questions**: give a paragraph-style **analysis** (who fits best and why). The UI shows CV cards separately, so the model does not repeat a numbered list in the text.

### Web UI layout

Each response with sources enabled is split into two parts:

1. **Assistant analysis** — the written recommendation (supports basic markdown such as **bold**).
2. **CVs behind this answer** — expandable cards with CV excerpts. A short callout explains that the **entire library was searched**, how many candidates were shortlisted, and (for ranking questions) which ones are featured in the analysis vs. shown as alternates.

For example, asking for the _top 3_ may still show **5** CV cards: the analysis focuses on 3, and 2 additional strong matches are available for comparison.

---

## Project structure

```
cv-rag-fastapi/
├── main.py              # FastAPI app, routes, serves the UI
├── rag.py               # Retrieval, context building, LLM chain
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image
├── .env                 # Secrets and config (create locally; never commit)
└── static/
    └── index.html       # Chat UI (HTML + JavaScript + Tailwind CDN)
```

| File                | Responsibility                                                 |
| ------------------- | -------------------------------------------------------------- |
| `main.py`           | HTTP API, loads RAG once at startup, mounts static files       |
| `rag.py`            | Pinecone retriever, prompt, LangChain chain, source formatting |
| `static/index.html` | Chat interface, streaming, source cards, loading states        |

---

## Installation

### 1. Clone and open the project

```powershell
cd cv-rag-fastapi
```

### 2. Create a virtual environment (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Python version:** Use **3.11**. On Windows, Python 3.13+ may try to compile NumPy from source and fail.
>
> If activation is blocked, run once (Current User):
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. Configure environment variables

Create a `.env` file in the project root (copy from a teammate or template). Example:

```env
PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key

PINECONE_INDEX=rag-rahaf
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
RETRIEVER_TOP_K=5
```

| Variable             | Description                                                                        |
| -------------------- | ---------------------------------------------------------------------------------- |
| `PINECONE_API_KEY`   | API key for your Pinecone project                                                  |
| `OPENROUTER_API_KEY` | API key for [OpenRouter](https://openrouter.ai/)                                   |
| `PINECONE_INDEX`     | Name of the index that holds your CV vectors                                       |
| `EMBEDDING_MODEL`    | Hugging Face model ID used to embed questions (must match how the index was built) |
| `OPENROUTER_MODEL`   | Chat model ID on OpenRouter                                                        |
| `RETRIEVER_TOP_K`    | Max number of distinct CVs returned per question                                   |

Never commit `.env` or share API keys in chat, screenshots, or git.

### 4. Run the server

```powershell
uvicorn main:app --reload
```

On startup you should see something like:

```
[rag] RAG chain ready. Index='rag-rahaf', Model='...', top_k=5
INFO:     Uvicorn running on http://127.0.0.1:8000
```

| URL                          | Description                             |
| ---------------------------- | --------------------------------------- |
| http://localhost:8000/       | Chat UI                                 |
| http://localhost:8000/docs   | Interactive API documentation (Swagger) |
| http://localhost:8000/health | Health check                            |

**Port in use?** Use another port: `uvicorn main:app --reload --port 8001`

After UI changes, hard-refresh the browser: `Ctrl+Shift+R`.

---

## Using the chat UI

Open http://localhost:8000/ in a browser.

- **Ask questions** in plain English (hiring, skills, comparisons, rankings).
- **Show sources** (toggle) — displays the analysis plus CV excerpt cards.
- **Stream response** (toggle) — answer appears word-by-word; turn off for a single JSON response.
- **Suggestion chips** — quick examples for common HR queries.
- **Status indicator** — shows whether the API is reachable.

The UI is static HTML served by FastAPI; there is no separate frontend build step.

---

## API reference

### `GET /health`

Returns `{"status": "ok"}` when the server is running.

### `POST /ask`

Full RAG pipeline — retrieval + LLM answer.

```powershell
$body = @{
  question     = "Which candidates know Python and machine learning?"
  show_context = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/ask `
  -Method Post -Body $body -ContentType "application/json"
```

| Field          | Type   | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `question`     | string | Required. The recruiter’s question.                          |
| `show_context` | bool   | If `true`, includes `context` and `sources` in the response. |

Example response (with `show_context: true`):

```json
{
  "question": "...",
  "answer": "...",
  "context": "...",
  "sources": [
    {
      "student_name": "Jane Doe",
      "doc_name": "cv_0012.pdf",
      "score": 0.82,
      "snippet": "...",
      "snippet_truncated": true,
      "rank": 1
    }
  ]
}
```

### `POST /sources`

Retrieval only — **no LLM call**. Fast way to preview which CVs match a question.

```powershell
$body = '{"question": "Rank the top 3 candidates for a data engineering role"}'
Invoke-RestMethod -Uri http://localhost:8000/sources `
  -Method Post -Body $body -ContentType "application/json"
```

Response includes `is_ranking` and `top_n` so the UI can label ranking questions correctly.

### `POST /ask-stream`

Same pipeline as `/ask`, but streams the answer as plain text tokens (used by the chat UI).

```powershell
curl -X POST http://localhost:8000/ask-stream `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Who is the best fit for a backend role?\"}"
```

---

## Docker

```powershell
docker build -t cv-rag-api .
docker run --rm -p 8000:8000 --env-file .env cv-rag-api
```

The image includes `static/`, so the chat UI works at http://localhost:8000/ inside the container.

---

## Architecture notes

- **Startup:** `init_rag()` in `rag.py` runs once when FastAPI starts. Pinecone, the embedding model, and the LangChain chain are reused for every request.
- **Chain:** Question → retriever (parallel) → format context → prompt → OpenRouter LLM → answer string.
- **Metadata:** Each Pinecone vector should include `text`, `doc_name`, and ideally `student_name` so the UI and answers can show real names.
- **CORS:** Enabled for all origins so the UI and external tools can call the API during development.

---

## Troubleshooting

| Issue                      | What to try                                                                   |
| -------------------------- | ----------------------------------------------------------------------------- |
| `PINECONE_API_KEY missing` | Add keys to `.env` in the project root; restart uvicorn.                      |
| Empty or weak answers      | Confirm `PINECONE_INDEX` and `EMBEDDING_MODEL` match the index you built.     |
| Wrong candidates           | Rephrase the question; increase `RETRIEVER_TOP_K` in `.env` (e.g. to 8).      |
| Stale UI after edits       | Hard refresh (`Ctrl+Shift+R`) — `index.html` is served with no-cache headers. |
| Port 8000 in use           | Stop the other process or use `--port 8001`.                                  |

---

## License & contributions

Use and adapt this project for learning and internal recruiting demos. Keep API keys private and do not commit `.env`.
