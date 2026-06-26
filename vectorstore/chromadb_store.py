import uuid
from types import SimpleNamespace
import chromadb
from chromadb.config import Settings


class ChromaDBVectorStore:
    """ChromaDB-based vector store for semantic search."""

    def __init__(
        self,
        db_path: str,
        collection_name: str = "investor-intelligence"
    ) -> None:
        """
        Initialize ChromaDB vector store.
        
        Args:
            db_path: Path to persistent ChromaDB storage.
            collection_name: Name of the collection to use.
        """
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to ChromaDB.
        
        Args:
            chunks: List of document chunks with page_content.
            embeddings: Embedding model (has embed_query method).
            company: Company name for metadata.
            year: Year for metadata.
            source_file: Source PDF filename.
        """
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append({
                "company": company,
                "year": str(year),
                "source_file": source_file
            })
            
            # Generate embedding for chunk
            embedding = embeddings.embed_query(chunk.page_content)
            embeddings_list.append(embedding)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

        print(f"Uploaded {len(documents)} chunks to ChromaDB.")


class Retriever:
    """Wrapper for retrieving relevant chunks from ChromaDB."""
    
    def __init__(self, collection):
        """
        Initialize retriever.
        
        Args:
            collection: ChromaDB collection instance.
        """
        self.collection = collection

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """
        Retrieve relevant chunks from ChromaDB.
        
        Args:
            query: Search query.
            company: Filter by company (optional).
            year: Filter by year (optional).
            top_k: Number of results to return.
            
        Returns:
            List of SimpleNamespace objects with page_content.
        """
        # Build where filter for metadata
        where_filter = None
        if company and year:
            where_filter = {
                "$and": [
                    {"company": {"$eq": company}},
                    {"year": {"$eq": str(year)}}
                ]
            }
        elif company:
            where_filter = {"company": {"$eq": company}}
        elif year:
            where_filter = {"year": {"$eq": str(year)}}

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        documents = []
        if results and results["documents"]:
            for doc_text in results["documents"][0]:
                documents.append(SimpleNamespace(page_content=doc_text))

        return documents
