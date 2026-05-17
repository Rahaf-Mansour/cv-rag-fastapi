"""
RAG core logic — extracted from the Module3-AddLLM notebook (cells 3–7).

Loaded ONCE at FastAPI startup via init_rag(); every HTTP request reuses the
same Pinecone connection, embedding model, and LCEL chain.

Each Pinecone chunk carries metadata={text, doc_name, student_name} (set
during indexing in Part 1), so the candidate's name is read directly from
the index — no runtime extraction needed.
"""

import os
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
class PineconeRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the Pinecone CV index from Part 1."""

    index: object
    embed_model: object
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        vector = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        response = self.index.query(
            vector=vector, top_k=self.top_k, include_metadata=True
        )

        docs: List[Document] = []
        for match in response["matches"]:
            md = match.get("metadata", {}) or {}
            # student_name may be missing or empty string for legacy chunks;
            # normalise to None so downstream code can treat it uniformly.
            student_name = md.get("student_name") or None
            doc = Document(
                page_content=md.get("text", ""),
                metadata={
                    "doc_name": md.get("doc_name", "unknown"),
                    "student_name": student_name,
                    "score": round(match["score"], 4),
                },
            )
            docs.append(doc)

        return docs


# ── Part B: Context formatter ────────────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved Documents into a single labelled context string.

    Each block is labelled with the candidate's name (from Pinecone metadata)
    so the LLM can cite people by name, not filename.
    """
    parts = []
    for doc in docs:
        doc_name = doc.metadata.get("doc_name", "unknown")
        student = doc.metadata.get("student_name") or doc_name
        header = (
            f"[Candidate: {student} | CV: {doc_name} "
            f"| Score: {doc.metadata.get('score', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n---\n".join(parts)


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
- When recommending or referencing candidates, ALWAYS cite them by their full name (the value of "Candidate" in each context block). You may add the CV filename in parentheses, e.g. "Lucas Walker (cv_007.pdf)".
- If a candidate's name is unknown, fall back to citing the CV filename.
- Be concise, factual, and structured in your responses.
- Do not hallucinate skills, experiences, or qualifications not explicitly mentioned in the context.

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

    # Cell 7: assemble the LCEL chain
    _rag_chain = (
        {
            "context": _retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(
        f"[rag] RAG chain ready. Index='{PINECONE_INDEX}', "
        f"Model='{OPENROUTER_MODEL}', top_k={RETRIEVER_TOP_K}"
    )


# ── Query helpers (replace the notebook's ask() function) ────────────────────
def _doc_to_source(d: Document) -> dict:
    """Convert a retrieved Document into a UI-friendly source dict."""
    return {
        "student_name": d.metadata.get("student_name"),
        "doc_name": d.metadata.get("doc_name", "unknown"),
        "score": d.metadata.get("score", 0.0),
        "snippet": (d.page_content or "")[:300],
    }


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
        result["context"] = format_docs(docs)
        result["sources"] = [_doc_to_source(d) for d in docs]

    return result


def run_query_stream(question: str):
    """Generator yielding answer tokens one at a time (for streaming endpoint)."""
    if _rag_chain is None:
        raise RuntimeError("RAG chain not initialised. Call init_rag() first.")

    for chunk in _rag_chain.stream(question):
        yield chunk


def get_sources(question: str) -> list[dict]:
    """Return the list of structured sources for a question (no LLM call)."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialised. Call init_rag() first.")

    docs = _retriever.get_relevant_documents(question)
    return [_doc_to_source(d) for d in docs]
