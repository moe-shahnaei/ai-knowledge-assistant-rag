from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "knowledge_base"


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def build_vector_database() -> None:
    chunks = load_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(documents).tolist()
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Saved {len(chunks)} chunks into Chroma vector database.")


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_results = []

    for index, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][index - 1]

        retrieved_results.append({
            "rank": index,
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "text": document
        })

    return retrieved_results

def main() -> None:
    build_vector_database()

    while True:
        query = input("\nAsk a question, or type 'exit': ")

        if query.lower() == "exit":
            break

        results = search_knowledge_base(query)

        print(f"\nQuestion: {query}\n")
        print("Top retrieved chunks:")

        for result in results:
            print("=" * 80)
            print(f"Result {result['rank']}")
            print(f"Source: {result['source']} | Chunk: {result['chunk_id']}")
            print("-" * 80)
            print(result["text"])
    
if __name__ == "__main__":
    main()