from rank_bm25 import BM25Okapi

from .chunkers import sentence_chunks
from .embeddings import EmbeddingModel
from .loaders import load_document
from .vector_store import VectorStore


def build_hybrid_retriever():
    document = load_document("data/raw/sample.txt")
    chunks = sentence_chunks(document)

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.embed_chunks(chunks)

    vector_store = VectorStore()
    vector_store.add(embedded_chunks)

    texts = [chunk.content for chunk in chunks]
    tokenized_texts = [
        text.lower().split()
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized_texts)

    return vector_store, embedding_model, bm25, chunks


def hybrid_search(
    query,
    vector_store,
    embedding_model,
    bm25,
    chunks,
    top_k=10,
    candidate_k=10,
):
    query_embedding = embedding_model.embed_query(query)

    dense_results = vector_store.search(
        query_embedding,
        top_k=candidate_k,
    )

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:candidate_k]

    rrf_scores = {}

    k = 60

    for rank, (item, score) in enumerate(
        dense_results,
        start=1,
    ):
        chunk_index = item.chunk.chunk_index

        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0)
            + 1 / (k + rank)
        )

    for rank, chunk_index in enumerate(
        bm25_indices,
        start=1,
    ):
        rrf_scores[chunk_index] = (
            rrf_scores.get(chunk_index, 0)
            + 1 / (k + rank)
        )

    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        (chunks[index], score)
        for index, score in ranked_chunks
    ]
