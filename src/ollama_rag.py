from ollama import Client

from .hybrid_retriever import build_hybrid_retriever, hybrid_search
from .citations import build_citation_context
from .citation_validator import validate_citation_numbers
from .memory import ConversationMemory

client = Client(host="http://localhost:11434")

def resolve_query(
        question:str,
        memory:ConversationMemory
)->str:
    history = memory.get_history()

    if not history:
        return question

    recent_history = history[-3:1]

    history_text = "\n".join(
        f"User: {turn['question']}\n"
        f"Assistant: {turn['answer']}"
        for turn in recent_history
    )

    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the latest user question as a "
                    "standalone search query using conversation "
                    "history when needed. Preserve the user's "
                    "intent. Do not answer the question. "
                    "If it is already self-contained, return it "
                    "unchanged. Return only the rewritten query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest question: {question}"
                ),
            },
        ],
        options={"temperature": 0},
    )

    resolved_query = response["message"]["content"].strip()
    return resolved_query or question

def generate_answer(
    question: str,
    memory: ConversationMemory,
) -> str:
    vector_store, embedding_model, bm25, chunks = (
        build_hybrid_retriever()
    )

    resolved_query = resolve_query(question, memory)

    print(f"\nOriginal question: {question}")
    print(f"Retrieval query: {resolved_query}")

    results = hybrid_search(
        resolved_query,
        vector_store,
        embedding_model,
        bm25,
        chunks,
        candidate_k=10,
        top_k=3,
    )

    print("\nRetrieved sources:")

    for citation_number, (item, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"[Source {citation_number}] "
            f"{item.source} | "
            f"Chunk {item.chunk_index} | "
            f"RRF Score {score:.4f}"
        )

    context, source_mapping = build_citation_context(results)

    history = memory.get_history()

    if history:
        history_text = "\n".join(
            f"Previous question: {turn['question']}\n"
            f"Previous answer: {turn['answer']}"
            for turn in history
        )
    else:
        history_text = "No previous conversation."

    prompt = f"""
Answer the current question using only the provided sources.

Conversation history:
{history_text}

Requirements:
- Give 2 to 3 clear sentences.
- Use only information supported by the provided sources.
- Use citations such as [Source 1] after factual claims.
- Only use source numbers that appear in the provided sources.
- Do not invent information.
- Use conversation history to understand follow-up questions.
- Treat conversation history as context, not as evidence.
- If the sources do not contain enough information, say:
"I don't have enough information in the provided context."

Sources:
{context}

Current question:
{question}

Answer:
"""

    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. "
                    "Answer using only the supplied sources. "
                    "Use conversation history to understand "
                    "the user's question, but cite the current "
                    "sources for factual claims."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={"temperature": 0},
    )

    FALLBACK_ANSWER = (
    "I don't have enough information in the provided context.")

    MIN_SIMILARITY = 0.35

    dense_results = vector_store.search(
    embedding_model.embed_query(resolved_query),
    top_k=1
)

    if not dense_results:
        memory.add_turn(question, FALLBACK_ANSWER)
        return FALLBACK_ANSWER

    best_similarity = dense_results[0][1]

    print(f"Best dense similarity: {best_similarity:.4f}")

    if best_similarity < MIN_SIMILARITY:
        print("Guardrail: Retrieved evidence may be insufficient.")
        memory.add_turn(question, FALLBACK_ANSWER)
        return FALLBACK_ANSWER    

    answer = response["message"]["content"]

    invalid_citations = validate_citation_numbers(
        answer,
        source_mapping,
    )

    if invalid_citations:
        print("\nWarning: Invalid citation numbers found:")
        print(invalid_citations)

    memory.add_turn(question, answer)

    return answer



if __name__ == "__main__":
    memory = ConversationMemory(max_turns=5)

    print("GroundTruth — Conversational RAG")
    print("Ask a question, or type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Conversation ended.")
            break

        if not question:
            continue

        answer = generate_answer(question, memory)

        print("\nGroundTruth:")
        print(answer)
