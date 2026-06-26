import os
from dotenv import load_dotenv
import chromadb

load_dotenv()


def create_chroma_index(db_path: str, collection_name: str = "investor-intelligence") -> None:
    """
    Initialize ChromaDB with persistent storage.

    Args:
        db_path: Path to persistent ChromaDB storage.
        collection_name: Name of the collection to create/initialize.
    """
    # Create directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Initialize persistent client
    client = chromadb.PersistentClient(path=db_path)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB collection '{collection_name}' initialized at {db_path}")


if __name__ == "__main__":
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_data")
    create_chroma_index(db_path=db_path)
