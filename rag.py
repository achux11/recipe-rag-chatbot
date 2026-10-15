from typing import List, Dict

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL_NAME,
    CHAT_MODEL_NAME,
)


# --- Safety checks for API key ---
if not OPENAI_API_KEY:
    print(
        "Warning: OPENAI_API_KEY is not set. "
        "Set it as an environment variable before running the app."
    )


_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

_chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(),
)
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _dedupe_docs(docs: List[Dict]) -> List[Dict]:
    """Remove exact/near-exact duplicates while preserving order."""
    seen = set()
    out: List[Dict] = []
    for d in docs:
        source = str((d.get("metadata") or {}).get("source", "unknown")).strip().lower()
        text = _normalize_ws(str(d.get("text", ""))).lower()
        key = (source, text)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _format_retrieval_only_answer(user_query: str, docs: List[Dict]) -> str:
    """
    Produce a readable answer without calling an LLM.

    Heuristics:
    - Deduplicate repeated chunks.
    - Prefer sources that match the query tokens (e.g. butter_chicken.txt).
    - Present the best source first; keep the rest as "snippets" for transparency.
    """
    docs = _dedupe_docs(docs)
    if not docs:
        return (
            "I couldn't find any relevant recipes in the knowledge base for that question. "
            "Try adding/ingesting more recipes into the `data` folder."
        )

    q = _normalize_ws(user_query).lower()
    q_tokens = {t for t in q.replace("-", " ").replace("_", " ").split() if t}

    def source_score(d: Dict) -> int:
        src = str((d.get("metadata") or {}).get("source", "")).lower()
        txt = str(d.get("text", "")).lower()
        s = 0
        if src and any(tok in src for tok in q_tokens):
            s += 5
        if q and (q in src or q in txt):
            s += 5
        # small boost for "title"/"ingredients"/"steps" chunks which read better standalone
        if "title:" in txt:
            s += 2
        if "ingredients" in txt:
            s += 1
        if "steps" in txt:
            s += 1
        return s

    docs_sorted = sorted(docs, key=source_score, reverse=True)
    best = docs_sorted[0]
    best_source = str((best.get("metadata") or {}).get("source", "unknown"))
    best_text = str(best.get("text", "")).strip()

    # Keep only snippets from the same source as the best match to avoid confusion
    same_source_snippets = []
    for d in docs_sorted:
        src = str((d.get("metadata") or {}).get("source", "unknown"))
        txt = str(d.get("text", "")).strip()
        if not txt:
            continue
        if src == best_source and txt != best_text:
            same_source_snippets.append(txt)

    extras = ""
    if same_source_snippets:
        extras = "\n\nMore from this recipe:\n\n" + "\n\n".join(same_source_snippets)

    return (
        "I can’t reach the language model right now (most commonly due to exceeded quota), "
        "so I’m answering using your recipe files directly.\n\n"
        f"**Best match: {best_source}**\n\n"
        f"{best_text}"
        f"{extras}"
    )


def embed_texts(texts: List[str]):
    """Return list of embedding vectors for given texts."""
    return _embedding_model.encode(texts).tolist()


def retrieve_recipes(query: str, k: int = 5) -> List[Dict]:
    """Retrieve top-k relevant recipe chunks from Chroma.

    Includes a lightweight keyword-based re-ranking step so that
    very specific queries like 'butter chicken' prefer the most
    obviously matching recipes (e.g. butter chicken) over more
    generic ones (e.g. any chicken dish).
    """
    try:
        query_vec = embed_texts([query])[0]
        results = _collection.query(
            query_embeddings=[query_vec],
            n_results=max(k, 10),  # get more and re-rank
        )
        docs: List[Dict] = []
        for doc, meta in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
        ):
            docs.append({"text": doc, "metadata": meta or {}})

        # --- Simple keyword-based re-ranking ---
        q_lower = query.lower()
        q_tokens = set(q_lower.split())

        def score(d: Dict) -> int:
            text = d["text"].lower()
            source = str(d["metadata"].get("source", "")).lower()
            s = 0
            # Exact phrase match gets a big boost
            if q_lower in text or q_lower in source:
                s += 10
            # Overlap of individual words for finer boosting
            for tok in q_tokens:
                if tok and (tok in text or tok in source):
                    s += 1
            return s

        docs.sort(key=score, reverse=True)
        return docs[:k]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []


def build_prompt(user_query: str, context_docs: List[Dict], history: List[Dict]) -> str:
    """
    Build a prompt for the LLM using:
    - last few conversation turns
    - retrieved recipe chunks as context
    """
    history_str = ""
    for turn in history[-5:]:
        history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    context_str = "\n\n".join(
        f"[Recipe Source: {d['metadata'].get('source', 'unknown')}]\n{d['text']}"
        for d in context_docs
    )

    prompt = f"""
You are a helpful recipe recommendation assistant.
You can suggest recipes, explain steps, and adapt recipes to ingredients people have.
Only use the information in the provided recipe context when possible.
If the answer is not in the context, say you are not sure and suggest general cooking advice.

Conversation so far:
{history_str}

User question:
{user_query}

Recipe context:
{context_str}

Answer clearly. When you reference a recipe, mention its source name.
"""
    return prompt.strip()


def generate_answer(user_query: str, history: List[Dict]) -> str:
    """
    Main RAG entry point:
    - Retrieve relevant recipe chunks
    - Build prompt with conversation history and context
    - Call LLM to generate final answer
    """
    docs = retrieve_recipes(user_query, k=6)
    if not docs:
        return (
            "I couldn't find any relevant recipes in the knowledge base for that question. "
            "Try asking about ingredients or specific dishes, or make sure you've ingested documents."
        )

    # If no LLM client (no API key or quota issues), fall back to a simple retrieval-only answer.
    if _client is None:
        return _format_retrieval_only_answer(user_query, docs)

    prompt = build_prompt(user_query, docs, history)

    try:
        response = _client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback: show retrieved context instead of a hard error
        print(f"LLM error: {e}")
        return (
            "There was an error generating a response from the language model "
            "(for example, insufficient quota / exceeded quota). Here’s what I found in your recipe files instead:\n\n"
            + _format_retrieval_only_answer(user_query, docs)
        )


