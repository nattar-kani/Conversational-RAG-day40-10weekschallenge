
from ollama import Client

from .citations import build_citation_context
from .hybrid_retriever import build_hybrid_retriever, hybrid_search
from .memory import ConversationMemory
from .ollama_rag import resolve_query


client = Client(host="http://localhost:11434")
MODEL_NAME = "qwen2.5:1.5b"

FALLBACK_ANSWER = (
    "I don't have enough information in the provided context."
)
MIN_SIMILARITY = 0.35


def prepare_answer(question: str, memory: ConversationMemory):
    """Retrieve evidence and prepare an answer or fallback."""

    vector_store, embedding_model, bm25, chunks = (
        build_hybrid_retriever()
    )

    resolved_query = resolve_query(question, memory)

    results = hybrid_search(
        resolved_query,
        vector_store,
        embedding_model,
        bm25,
        chunks,
        candidate_k=10,
        top_k=3,
    )

    dense_results = vector_store.search(
        embedding_model.embed_query(resolved_query),
        top_k=1,
    )

    if not dense_results:
        return {
            "fallback": True,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    best_similarity = dense_results[0][1]

    if best_similarity < MIN_SIMILARITY or not results:
        return {
            "fallback": True,
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

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
- Include citations such as [Source 1] after factual claims.
- Only use source numbers provided in the sources below.
- Do not invent information.
- Use history to understand follow-up questions, not as evidence.
- If the sources do not contain enough information, say:
"{FALLBACK_ANSWER}"

Sources:
{context}

Current question:
{question}

Answer:
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are GroundTruth, a helpful RAG assistant. "
                "Answer using only the supplied sources. "
                "Cite sources for factual claims."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    sources = [
        {
            "number": number,
            **details,
        }
        for number, details in source_mapping.items()
    ]

    return {
        "fallback": False,
        "messages": messages,
        "sources": sources,
        "resolved_query": resolved_query,
        "best_similarity": best_similarity,
    }


def stream_answer(prepared, question, memory):
    """Yield answer tokens and save the completed answer to memory."""

    response_stream = client.chat(
        model=MODEL_NAME,
        messages=prepared["messages"],
        options={"temperature": 0},
        stream=True,
    )

    answer_parts = []

    for chunk in response_stream:
        token = chunk["message"]["content"]

        if token:
            answer_parts.append(token)
            yield token

    answer = "".join(answer_parts).strip()

    if answer:
        memory.add_turn(question, answer)
