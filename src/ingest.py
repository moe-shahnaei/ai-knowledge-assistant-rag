from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


def load_text_files(folder_path: Path) -> list[dict]:
    documents = []

    for file_path in folder_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Split text into cleaner chunks using paragraphs and sentences.
    This avoids cutting words in the middle like character-based chunking.
    """

    text = text.replace("\r\n", "\n").strip()

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    sentences = []

    for paragraph in paragraphs:
        parts = paragraph.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n")
        for part in parts:
            sentence = part.strip()
            if sentence:
                sentences.append(sentence)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_text = " ".join(current_chunk)

        if len(current_text) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            if overlap_sentences > 0:
                current_chunk = current_chunk[-overlap_sentences:]
            else:
                current_chunk = []

            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_chunks(documents: list[dict]) -> list[dict]:
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "id": f"{document['source']}_chunk_{index}",
                "source": document["source"],
                "chunk_id": index,
                "text": chunk
            })

    return all_chunks


def save_chunks(chunks: list[dict], output_file: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")


def main() -> None:
    documents = load_text_files(DOCS_DIR)
    chunks = create_chunks(documents)
    save_chunks(chunks, OUTPUT_FILE)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Saved chunks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()