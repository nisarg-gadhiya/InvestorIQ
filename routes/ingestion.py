import shutil
from fastapi import APIRouter, File, UploadFile
from pathlib import Path
import os
from llm.gemini import get_embedding_model
from vectorstore.chromadb_store import ChromaDBVectorStore
from ingestion.ingest_documents import ingest_document

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

        # Initialize embeddings and vector store
        embeddings = get_embedding_model()

        vector_store = ChromaDBVectorStore(
            db_path=os.getenv("CHROMA_DB_PATH", "./chroma_data")
        )

        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store
        )

    return {
        "message": "Document uploaded successfully",
        "file_name": file.filename
    }