import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vectorstore.chromadb_store import ChromaDBVectorStore, Retriever
from llm.openai import get_openai_client

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None

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

        # Build chat prompt – include retrieved context and the user question
        prompt = f"You are an expert financial analyst. Use the following context from corporate reports to answer the user's question. If the context does not contain relevant information, politely indicate that you do not have enough data.\n\nContext:\n{context}\n\nUser Question: {request.question}\n\nAnswer:"

        client = get_openai_client()
        response = client.generate_content(prompt)
        answer = response.text
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
