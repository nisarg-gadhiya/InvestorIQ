import os
import sys
from dotenv import load_dotenv

load_dotenv()

from vectorstore.chromadb_store import ChromaDBVectorStore


def search_vectorstore(query: str, top: int = 5):
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_data")

    if not db_path:
        raise RuntimeError(
            "Missing ChromaDB configuration. Set CHROMA_DB_PATH in your .env."
        )

    store = ChromaDBVectorStore(db_path=db_path)
    results = store.collection.query(query_texts=[query], n_results=top)

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results.get('documents', [[]])[0])}\n")

    if results.get('documents') and results['documents'][0]:
        for idx, doc_content in enumerate(results['documents'][0], start=1):
            snippet = doc_content.strip().replace("\n", " ") if isinstance(doc_content, str) else "<no content>"
            if len(snippet) > 350:
                snippet = snippet[:350].rstrip() + "..."

            print(f"Result {idx}")
            print(f"  content snippet: {snippet}")
            print("  " + "-" * 60)
    else:
        print("No results returned. Verify your collection contents or try a different query.")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieval_debug \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()
