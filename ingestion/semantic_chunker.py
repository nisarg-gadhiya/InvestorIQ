import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv
load_dotenv()



def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: Embedding model with embed_query method.

    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = splitter.create_documents([markdown_content])

    for chunk in chunks:
        raw_content = chunk.page_content
        marker_positions = []
        for marker in re.finditer(r"<!--\s*PAGE\s*(\d+)\s*-->", raw_content, flags=re.IGNORECASE):
            marker_positions.append((marker.start(), int(marker.group(1))))

        if marker_positions:
            chunk.metadata["page"] = marker_positions[-1][1]
        else:
            chunk.metadata["page"] = 1

        chunk.page_content = re.sub(r"<!--\s*PAGE\s*\d+\s*-->", "", raw_content, flags=re.IGNORECASE).strip()

    return chunks

if __name__ == "__main__":
    import os
    from llm.openai import get_embedding_model

    embeddings = get_embedding_model()

    markdown_file = "../data/markdown/2024_Apple.md"

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()