# CV RAG — FastAPI + Web UI

A FastAPI service that wraps the Module 3 notebook (`Module3-AddLLM-STUDENT.ipynb`) into a long-running web app **with a chat-style frontend**.

The Pinecone index built in **Part 1** (`rag-rahaf`) is reused; this app adds the LLM, REST API, and UI from **Part 2**.

## Project structure

```
cv-rag-fastapi/
├── .env              ← API keys + config (NEVER commit)
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── main.py           ← FastAPI routes + UI mount
├── rag.py            ← RAG logic extracted from the notebook
└── static/
    └── index.html    ← Chat-style web UI (Tailwind, streaming)
```

## Notebook → file mapping

| Notebook cell                   | Lives in                                                                |
| ------------------------------- | ----------------------------------------------------------------------- |
| Cell 1–2: imports & config      | `rag.py` top — `os.getenv()` replaces hardcoded values                  |
| Cell 3: Pinecone + embed model  | `rag.py` → `init_rag()`                                                 |
| Cell 4A: `PineconeRetriever`    | `rag.py` → class definition                                             |
| Cell 4B: `format_docs()`        | `rag.py` → standalone function                                          |
| Cell 5: prompt template         | `rag.py` → inside `init_rag()`                                          |
| Cell 6: LLM (OpenRouter)        | `rag.py` → inside `init_rag()`                                          |
| Cell 7: `rag_chain`             | `rag.py` → inside `init_rag()`                                          |
| Cell 8: `ask()`                 | `run_query()` + `run_query_stream()` in `rag.py`, called from `main.py` |
| (new) chat UI                   | `static/index.html` served at `/`                                       |

## 1. Set up environment

From inside `cv-rag-fastapi/`:

```powershell
# Create and activate a virtual env (Windows / PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> Use **Python 3.11**. Newer versions (3.13) need to compile NumPy from source on Windows.
>
> If activation is blocked once, run as admin:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. Configure `.env`

The repo ships a `.env` that already points at:

- Pinecone index: `rag-rahaf`
- Embedding model: `BAAI/bge-small-en-v1.5`
- OpenRouter model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
- `RETRIEVER_TOP_K=5`

Replace the API keys with your own — never commit them.

## 3. Run locally

```powershell
uvicorn main:app --reload
```

You should see:

```
[rag] RAG chain ready. Index='rag-rahaf', Model='nvidia/...', top_k=5
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

| URL                                 | What it is                              |
| ----------------------------------- | --------------------------------------- |
| <http://localhost:8000/>            | **Chat UI** — ask questions like a chat |
| <http://localhost:8000/docs>        | Swagger / OpenAPI playground            |
| <http://localhost:8000/health>      | Health check                            |

> **Port already in use?** Run `uvicorn main:app --reload --port 8001` and use that port everywhere below.

## 4. The Chat UI

Open <http://localhost:8000/> in any browser.

Features:

- **Chat-style layout** with user / assistant bubbles and streaming responses.
- **Live status pill** showing whether the API is online.
- **Suggestion chips** for common HR queries.
- **Source toggle** — every answer shows the retrieved CV chunks it relied on, expandable inline. This is the visual proof that the LLM is grounded in real CVs.
- **Stream toggle** — turn off if you prefer the full answer in one shot.

The UI uses Tailwind via CDN, so no build step is needed — it's pure HTML + JS served by FastAPI.

## 5. REST endpoints

### `GET /health`

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### `POST /ask`

Returns a JSON answer. Set `show_context: true` to also include the structured sources.

```powershell
$body = @{
  question     = "Which candidates know Python and ML?"
  show_context = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/ask `
  -Method Post -Body $body -ContentType "application/json"
```

Response shape:

```json
{
  "question": "...",
  "answer":   "...",
  "context":  "labelled context string (optional)",
  "sources":  [
    { "doc_name": "cv_007.pdf", "score": 0.731, "snippet": "..." }
  ]
}
```

### `POST /sources`

Just the retrieved CVs — no LLM call (fast, useful for previews):

```powershell
$body = '{"question":"Python developer"}'
Invoke-RestMethod -Uri http://localhost:8000/sources `
  -Method Post -Body $body -ContentType "application/json"
```

### `POST /ask-stream`

Streams the answer token-by-token as plain text. Used by the chat UI.

```powershell
curl -X POST http://localhost:8000/ask-stream `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Who is the best fit for a backend role?\"}"
```

## 6. Run with Docker

```powershell
docker build -t cv-rag-api .
docker run --rm -p 8000:8000 --env-file .env cv-rag-api
```

The image bundles `static/`, so the UI works inside the container too — just open <http://localhost:8000/>.

## How it differs from the notebook

| Notebook                                         | FastAPI app                                       |
| ------------------------------------------------ | ------------------------------------------------- |
| Cells run top-to-bottom **once** per session     | App runs **forever**, serves many requests        |
| Models reload every time you re-run the notebook | `init_rag()` runs **once** at startup (lifespan)  |
| Hardcoded API keys in cell 2                     | All secrets read from `.env` via `python-dotenv`  |
| `ask()` prints output to the cell                | `/ask` returns JSON, `/ask-stream` streams tokens |
| No UI — only Python cells                        | Full chat UI at `/` with source citations         |
