from retrieve import build_vector_database, search_knowledge_base


def create_grounded_answer(query: str, retrieved_results: list[dict]) -> str:
    if not retrieved_results:
        return "I do not have enough information in the knowledge base to answer this question."

    context = "\n\n".join(result["text"] for result in retrieved_results)

    answer = (
        "Based on the retrieved knowledge base content, here is a grounded answer:\n\n"
        f"{context}\n\n"
        "This answer is based only on the retrieved sources listed below."
    )

    return answer


def display_sources(retrieved_results: list[dict]) -> None:
    print("\nSources:")

    seen_sources = set()

    for result in retrieved_results:
        source_label = f"{result['source']} | chunk {result['chunk_id']}"

        if source_label not in seen_sources:
            print(f"- {source_label}")
            seen_sources.add(source_label)


def main() -> None:
    build_vector_database()

    print("\nAI Knowledge Assistant with RAG")
    print("Ask a question about the knowledge base.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nQuestion: ")

        if query.lower() == "exit":
            break

        retrieved_results = search_knowledge_base(query, top_k=2)
        answer = create_grounded_answer(query, retrieved_results)

        print("\nAnswer:")
        print(answer)

        display_sources(retrieved_results)


if __name__ == "__main__":
    main()