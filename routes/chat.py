import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from vectorstore.chromadb_store import ChromaDBVectorStore, Retriever
from llm.openai import get_openai_client

router = APIRouter()

# Similarity threshold — below this score, retrieval is considered weak
# ChromaDB cosine similarity: 1.0 = identical, 0.0 = completely different
SIMILARITY_THRESHOLD = 0.45

# Rephrased query template — used on retry to cast a wider semantic net
REPHRASE_TEMPLATE = (
    "financial data analysis {question} annual report metrics "
    "performance results figures statements"
)


def rephrase_query(question: str) -> str:
    """
    Rephrase the user's question into a broader financial search query
    for the retry attempt. Expands keywords without changing meaning.
    """
    return REPHRASE_TEMPLATE.format(question=question)


def build_not_found_response(question: str, company: str | None, year: int | None) -> dict:
    """
    Build an honest 'not found' response when retrieval fails after retry.
    No LLM call — deterministic, always accurate, never hallucinated.
    """
    company_context = ""
    if company and year:
        company_context = f" in the {year} {company} annual report"
    elif company:
        company_context = f" in the {company} annual report"

    stop_words = {
        "what", "was", "the", "is", "are", "were", "how", "did",
        "does", "do", "a", "an", "of", "in", "for", "and", "or",
        "tell", "me", "about", "can", "you", "please", "give", "find"
    }
    words = question.lower().replace("?", "").replace(",", "").split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    keyword_str = ", ".join(keywords[:4]) if keywords else "the relevant section"

    answer = (
        f"I couldn't find sufficient information about this{company_context}. "
        f"The uploaded document may not cover this topic directly, or it may be "
        f"described using different terminology.\n\n"
        f"**Suggested keywords to search manually:** {keyword_str}\n\n"
        f"If you believe this information should be in the report, try rephrasing "
        f"your question using different financial terms."
    )

    return {
        "answer": answer,
        "sources": [],
        "retrieval_status": "not_found"
    }


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("role must be either 'user' or 'assistant'")
        return value


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None
    history: list[Message] = []


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize vector store and retriever
        vector_store = ChromaDBVectorStore(
            db_path=os.getenv("CHROMA_DB_PATH", "./chroma_data")
        )
        retriever = Retriever(vector_store.collection)

        # ── Step 1: First retrieval attempt with original question ──
        if request.company and request.year:
            docs = retriever.invoke(
                query=request.question,
                company=request.company,
                year=request.year
            )
        else:
            docs = retriever.invoke(query=request.question)

        best_score = retriever.get_best_similarity(docs)

        # ── Step 2: If similarity is weak, retry with rephrased query ──
        if best_score < SIMILARITY_THRESHOLD:
            rephrased = rephrase_query(request.question)

            if request.company and request.year:
                retry_docs = retriever.invoke(
                    query=rephrased,
                    company=request.company,
                    year=request.year
                )
            else:
                retry_docs = retriever.invoke(query=rephrased)

            retry_score = retriever.get_best_similarity(retry_docs)

            if retry_score > best_score:
                docs = retry_docs
                best_score = retry_score

        # ── Step 3: If still below threshold after retry, return not-found ──
        if best_score < SIMILARITY_THRESHOLD:
            return build_not_found_response(
                question=request.question,
                company=request.company,
                year=request.year
            )

        # ── Step 4: Build sources list from retrieved docs ──
        sources = []
        seen_sources = set()
        for doc in docs:
            source_key = (doc.source_file, doc.page)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "source_file": doc.source_file,
                    "page": doc.page
                })

        # ── Step 5: Build context string from retrieved docs ──
        context = "\n\n".join(doc.page_content for doc in docs)

        # ── Step 6: Build messages with conversation history ──
        recent_history = request.history[-10:]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst. Answer the user's question using ONLY the context provided below from the company's annual report. "
                    "Do NOT use your training knowledge or any information outside this context. "
                    "If the context does not fully answer the question, say so clearly and answer only what the context supports. "
                    "Never fabricate figures, dates, or facts.\n\n"
                    f"Context from annual report:\n{context}"
                ),
            }
        ]

        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": request.question})

        # ── Step 7: Call LLM with grounded context ──
        client = get_openai_client()
        response = client.generate_content(messages=messages)
        answer = response.text

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_status": "found"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
