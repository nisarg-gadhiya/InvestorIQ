import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from vectorstore.chromadb_store import ChromaDBVectorStore, Retriever
from llm.openai import get_openai_client

router = APIRouter()


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

        # Retrieve relevant context
        context = ""
        if request.company and request.year:
            docs = retriever.invoke(
                query=request.question,
                company=request.company,
                year=request.year
            )
        else:
            docs = retriever.invoke(
                query=request.question
            )
        context = "\n\n".join(doc.page_content for doc in docs)

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

        recent_history = request.history[-10:]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst. Use the following context from "
                    "corporate reports to answer the user's question. If the context does "
                    "not contain relevant information, politely indicate that you do not "
                    "have enough data.\n\nContext:\n"
                    f"{context}"
                ),
            }
        ]

        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": request.question})

        client = get_openai_client()
        response = client.generate_content(messages=messages)
        answer = response.text
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
