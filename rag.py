"""
RAG core logic — extracted from the Module3-AddLLM notebook (cells 3–7).

Loaded ONCE at FastAPI startup via init_rag(); every HTTP request reuses the
same Pinecone connection, embedding model, and LCEL chain.

Each Pinecone chunk carries metadata={text, doc_name, student_name} (set
during indexing in Part 1), so the candidate's name is read directly from
the index — no runtime extraction needed.
"""

import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# LangChain core
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# ── Compatibility shim (same one used in the notebook) ───────────────────────
# Some langchain==0.1.20 + langchain-core combos try to read attributes that
# don't exist on the `langchain` module. Patch them defensively so imports
# don't blow up at runtime.
import langchain as _lc

_MISSING_ATTRS = ["debug", "verbose", "llm_cache", "callback_manager"]
for _attr in _MISSING_ATTRS:
    if not hasattr(_lc, _attr):
        setattr(_lc, _attr, None if _attr == "llm_cache" else False)


def _langchain_getattr(name):
    return False


_lc.__getattr__ = _langchain_getattr


load_dotenv()

# ── Config (read from .env) ──────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cv-rag")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free"
)
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

# ── Module-level singletons (populated by init_rag) ──────────────────────────
_retriever: Optional["PineconeRetriever"] = None
_rag_chain = None


# ── Part A: Custom LangChain Retriever ───────────────────────────────────────
# Minimum snippet length (in characters, after stripping) for a chunk to be
# considered "meaningful". Pinecone chunks that contain only section headers
# (e.g. "PROJECTS\nCertificates") add noise to both the LLM prompt and the UI,
# so we filter them out below.
MIN_SNIPPET_CHARS = 30

# CV section titles seen in indexed chunks (Part 1 indexing).
_SECTION_HEADER_RE = re.compile(
    r"^(?:PROJECTS?|EXPERIENCE|EDUCATION|CERTIFICATES?|"
    r"TECHNICAL\s+SKILLS?|SKILLS?|SUMMARY|OBJECTIVE|"
    r"WORK\s+EXPERIENCE|EMPLOYMENT|LANGUAGES?|REFERENCES?|"
    r"CONTACT|PROFILE|ABOUT|AWARDS?|ACHIEVEMENTS?|"
    r"PUBLICATIONS?|INTERESTS?|HOBBIES)(?:\s*[:\-–—])?\s*$",
    re.IGNORECASE,
)


def _looks_like_section_header(line: str) -> bool:
    """True when a line is a standalone CV section title, not body text."""
    s = line.strip()
    if not s or len(s) > 60:
        return False
    if _SECTION_HEADER_RE.match(s):
        return True
    # Short ALL-CAPS lines (e.g. "PROJECTS", "SKILLS") without bullets.
    words = s.split()
    if (
        s.isupper()
        and 1 <= len(words) <= 5
        and len(s) >= 3
        and not s.startswith(("-", "•", "*", "·"))
    ):
        return True
    return False


def _strip_empty_sections(text: str) -> str:
    """Remove section headers that have no body before the next header.

    Example input::

        PROJECTS
        Certificates
         Udacity Aug 2025
        - Intro to ML...

    becomes::

        Certificates
         Udacity Aug 2025
        - Intro to ML...
    """
    if not text or not text.strip():
        return ""

    lines = text.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str, list[str]]] = []
    preamble: list[str] = []
    current_header: Optional[str] = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_body
        if current_header is not None:
            sections.append((current_header, current_body))
        elif current_body:
            preamble.extend(current_body)
        current_header = None
        current_body = []

    for line in lines:
        if _looks_like_section_header(line):
            flush()
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line)

    flush()

    out: list[str] = list(preamble)
    for header, body in sections:
        body_text = "\n".join(body).strip()
        if body_text:
            out.append(header)
            out.extend(body)

    result = "\n".join(out).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def _cleaned_content(text: str) -> str:
    return _strip_empty_sections((text or "").strip())


def _content_quality(text: str) -> int:
    """Higher = more substantive text after removing empty sections."""
    return len(_cleaned_content(text))


def _is_meaningful(text: str) -> bool:
    cleaned = _cleaned_content(text)
    return bool(cleaned) and len(cleaned) >= MIN_SNIPPET_CHARS


class PineconeRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the Pinecone CV index from Part 1.

    Behaviour:
      * Queries Pinecone for ``top_k * pool_multiplier`` raw chunks.
      * Drops chunks whose text is too short to be useful (header-only chunks).
      * Deduplicates so each CV (``doc_name``) appears at most once, keeping
        the highest-scoring meaningful chunk per CV.
      * Returns up to ``top_k`` distinct candidates, sorted by score desc.

    This guarantees the LLM and the UI both see one entry per candidate and
    never receive empty header-only fragments.
    """

    index: object
    embed_model: object
    top_k: int = 5
    pool_multiplier: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _query_pool(self, query: str, k: int) -> List[Document]:
        """Run the raw Pinecone query and wrap matches as LangChain Documents."""
        vector = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        response = self.index.query(
            vector=vector, top_k=k, include_metadata=True
        )

        docs: List[Document] = []
        for match in response["matches"]:
            md = match.get("metadata", {}) or {}
            # student_name may be missing or empty string for legacy chunks;
            # normalise to None so downstream code can treat it uniformly.
            student_name = md.get("student_name") or None
            docs.append(
                Document(
                    page_content=md.get("text", ""),
                    metadata={
                        "doc_name": md.get("doc_name", "unknown"),
                        "student_name": student_name,
                        "score": round(match["score"], 4),
                    },
                )
            )
        return docs

    def _dedupe_by_cv(self, docs: List[Document]) -> List[Document]:
        """Keep at most one chunk per ``doc_name``.

        Selection rule, in order of preference:
          1. Prefer chunks that pass ``_is_meaningful`` over header-only ones.
          2. Among ties, keep the higher Pinecone score.
        """
        by_doc: dict = {}
        for d in docs:
            doc_name = d.metadata.get("doc_name", "unknown")
            existing = by_doc.get(doc_name)
            if existing is None:
                by_doc[doc_name] = d
                continue

            d_ok = _is_meaningful(d.page_content)
            ex_ok = _is_meaningful(existing.page_content)
            if d_ok and not ex_ok:
                by_doc[doc_name] = d
            elif not d_ok and ex_ok:
                pass
            else:
                d_q = _content_quality(d.page_content)
                ex_q = _content_quality(existing.page_content)
                if d_q > ex_q:
                    by_doc[doc_name] = d
                elif d_q == ex_q and (
                    d.metadata.get("score", 0) > existing.metadata.get("score", 0)
                ):
                    by_doc[doc_name] = d

        unique = list(by_doc.values())
        # Drop CVs whose only retrieved chunk is empty/header-only. If that
        # would remove everything, fall back to whatever we have so the UI
        # never goes silent.
        meaningful = [d for d in unique if _is_meaningful(d.page_content)]
        if meaningful:
            unique = meaningful

        unique.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return unique

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if _is_ranking_question(query):
            pool_k = max(self.top_k * self.pool_multiplier * 2, 25)
        else:
            pool_k = max(self.top_k * self.pool_multiplier, 15)
        raw = self._query_pool(query, pool_k)
        deduped = self._dedupe_by_cv(raw)
        return deduped[: self.top_k]


# ── Part B: Context formatter ────────────────────────────────────────────────
_RANKING_QUESTION_RE = re.compile(
    r"\b(rank|ranking|top\s*\d+|best\s*\d+|list\s+(?:the\s+)?top)\b",
    re.IGNORECASE,
)


def _is_ranking_question(question: str) -> bool:
    return bool(_RANKING_QUESTION_RE.search(question or ""))


def _extract_top_n(question: str, default: int = 3) -> int:
    m = re.search(r"\btop\s*(\d+)\b", question or "", re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"\bbest\s*(\d+)\b", question or "", re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    m = re.search(r"\brank\s+(?:the\s+)?top\s*(\d+)\b", question or "", re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    return default


def _meaningful_docs(docs: List[Document]) -> List[Document]:
    return [d for d in docs if _is_meaningful(d.page_content)]


def _ranking_context_note(docs: List[Document], question: str) -> str:
    """Tell the LLM who was retrieved — the UI shows CV excerpts, not a numbered answer."""
    n = _extract_top_n(question)
    lines = [
        "NOTE: The app shows CV excerpts below your analysis. Do NOT output a numbered list.",
        "Write an Analysis in paragraphs: who fits best, strengths, gaps. Use candidate names only.",
        "Do NOT write 'Rank #1', 'match 1', 'search score', or 'relevance' in your answer.",
        "",
        "CVs pulled for this question (for your reference only):",
    ]
    for doc in _meaningful_docs(docs)[:n]:
        doc_name = doc.metadata.get("doc_name", "unknown")
        student = doc.metadata.get("student_name") or doc_name
        lines.append(f"  • {student} ({doc_name})")
    return "\n".join(lines)


def format_docs(docs: List[Document], question: str = "") -> str:
    """Concatenate retrieved Documents into a single labelled context string.

    Blocks are ordered as returned from the CV index (internal use only).
    """
    parts = []
    for doc in _meaningful_docs(docs):
        doc_name = doc.metadata.get("doc_name", "unknown")
        student = doc.metadata.get("student_name") or doc_name
        header = f"[Candidate: {student} | CV: {doc_name}]"
        body = _cleaned_content(doc.page_content)
        parts.append(f"{header}\n{body}")

    if not parts:
        return "(no relevant CV sections were found)"

    body = "\n---\n".join(parts)
    if question and _is_ranking_question(question):
        return _ranking_context_note(docs, question) + "\n\n---- CV SECTIONS ----\n" + body
    return body


def _prepare_rag_inputs(payload: dict) -> dict:
    """Build prompt variables from retrieved docs + the user question."""
    return {
        "context": format_docs(payload["docs"], payload["question"]),
        "question": payload["question"],
    }


# ── Startup: load models and build chain ─────────────────────────────────────
def init_rag() -> None:
    """
    Run ONCE at FastAPI startup.

    Connects to Pinecone, loads the embedding model, and assembles the LCEL
    chain. The retriever and chain are stored as module-level singletons so
    every HTTP request reuses them.
    """
    global _retriever, _rag_chain

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing — check your .env file.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing — check your .env file.")

    # Cell 3: connect to Pinecone + load embedding model
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    _retriever = PineconeRetriever(
        index=index,
        embed_model=embed_model,
        top_k=RETRIEVER_TOP_K,
    )

    # Cell 5: prompt template
    SYSTEM_TEMPLATE = """You are an expert technical recruiter assistant specializing in evaluating and matching candidates from a CV database.

RULES:
- Base your answer ONLY on the provided CV context below. Do not use external knowledge or make assumptions.
- If the context does not contain sufficient information to answer the question, say explicitly: "The provided CVs do not contain enough information to answer this question."
- When recommending or referencing candidates, cite them by full name (from "Candidate" in each block). Add the CV filename in parentheses when helpful, e.g. "Lucas Walker (cv_007.pdf)".
- If a candidate's name is unknown, use the CV filename.
- In user-facing text, use candidate names only—never mention "Rank #", "match number", "search score", or "relevance".
- Be concise, factual, and structured. Do not hallucinate skills or experience not in the context.
- Only discuss candidates who appear in the context blocks.

RANKING / TOP-N QUESTIONS (when the context says CV excerpts are shown separately):
- Do NOT output a numbered ranking list (the app shows CV cards below your analysis).
- Write an **Analysis** in short paragraphs: who you recommend for the role and why, key skills, notable gaps.
- You may prefer a candidate whose CV was found later in the list—explain in plain HR language (no technical ranking terms).
- Do not invent candidates not in the context.

GENERAL QUESTIONS:
- Answer directly in clear prose. Use bullet points only when they improve readability.

---- CV CONTEXT ----
{context}
--------------------
"""
    HUMAN_TEMPLATE = "{question}"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ]
    )

    # Cell 6: LLM via OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name=OPENROUTER_MODEL,
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CV-RAG-FastAPI",
        },
    )

    # Cell 7: assemble the LCEL chain (retriever + question → ranked context)
    _rag_chain = (
        RunnableParallel(docs=_retriever, question=RunnablePassthrough())
        | RunnableLambda(_prepare_rag_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    print(
        f"[rag] RAG chain ready. Index='{PINECONE_INDEX}', "
        f"Model='{OPENROUTER_MODEL}', top_k={RETRIEVER_TOP_K}"
    )


# ── Query helpers (replace the notebook's ask() function) ────────────────────
SNIPPET_CHARS = 500


def _doc_to_source(d: Document, rank: int = 0) -> dict:
    """Convert a retrieved Document into a UI-friendly source dict."""
    cleaned = _cleaned_content(d.page_content)
    snippet = cleaned[:SNIPPET_CHARS]
    out = {
        "student_name": d.metadata.get("student_name"),
        "doc_name": d.metadata.get("doc_name", "unknown"),
        "score": d.metadata.get("score", 0.0),
        "snippet": snippet,
        "snippet_truncated": len(cleaned) > SNIPPET_CHARS,
    }
    if rank:
        out["rank"] = rank
    return out


def run_query(question: str, show_context: bool = False) -> dict:
    """
    Run the RAG chain and return a dict with the answer.

    When show_context=True the response also includes:
        - context: the same labelled string used in the LLM prompt (debug view)
        - sources: a structured list of {student_name, doc_name, score, snippet}
    """
    if _rag_chain is None or _retriever is None:
        raise RuntimeError("RAG chain not initialised. Call init_rag() first.")

    answer = _rag_chain.invoke(question)
    result = {"question": question, "answer": answer}

    if show_context:
        docs = _retriever.get_relevant_documents(question)
        result["context"] = format_docs(docs, question)
        ranked = _meaningful_docs(docs)
        result["sources"] = [
            _doc_to_source(d, rank=i) for i, d in enumerate(ranked, start=1)
        ]

    return result


def _friendly_llm_error(exc: Exception) -> str:
    """Turn provider errors into short messages for the chat UI."""
    err = str(exc)
    name = type(exc).__name__
    if "429" in err or "RateLimit" in name or "rate limit" in err.lower():
        return (
            "The AI provider rate limit was reached (OpenRouter free tier). "
            "Wait for the daily reset, add credits at https://openrouter.ai, "
            "or change OPENROUTER_MODEL in your .env to a model with quota left."
        )
    if "401" in err or "authentication" in err.lower() or "invalid" in err.lower():
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in your .env file."
    return f"Could not generate an answer: {err[:300]}"


def run_query_stream(question: str):
    """Generator yielding answer tokens one at a time (for streaming endpoint)."""
    if _rag_chain is None:
        raise RuntimeError("RAG chain not initialised. Call init_rag() first.")

    try:
        for chunk in _rag_chain.stream(question):
            yield chunk
    except Exception as e:
        yield _friendly_llm_error(e)


def get_sources_payload(question: str) -> dict:
    """Return structured sources plus metadata for the UI (no LLM call)."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialised. Call init_rag() first.")

    docs = _retriever.get_relevant_documents(question)
    is_ranking = _is_ranking_question(question)
    return {
        "sources": [
            _doc_to_source(d, rank=i)
            for i, d in enumerate(_meaningful_docs(docs), start=1)
        ],
        "is_ranking": is_ranking,
        "top_n": _extract_top_n(question) if is_ranking else None,
    }


def get_sources(question: str) -> list[dict]:
    """Return the list of structured sources for a question (no LLM call)."""
    return get_sources_payload(question)["sources"]
