# src/ingest.py

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"


def load_text_files(folder_path: Path) -> list[dict]:
    """
    Load all .txt files from the sample_docs folder.
    Each document is stored with its filename and text content.
    """
    documents = []

    for file_path in folder_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into smaller overlapping chunks.
    Overlap helps preserve context between chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(documents: list[dict]) -> list[dict]:
    """
    Create chunks for each loaded document and keep track of the source file.
    """
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": chunk
            })

    return all_chunks


def main() -> None:
    documents = load_text_files(DOCS_DIR)
    chunks = create_chunks(documents)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} text chunks.\n")

    for chunk in chunks:
        print("=" * 80)
        print(f"Source: {chunk['source']} | Chunk: {chunk['chunk_id']}")
        print("-" * 80)
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()