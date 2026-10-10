import faiss
import numpy as np

from .models import EmbeddedChunk
from .vector_store_base import BaseVectorStore

class VectorStore(BaseVectorStore):
    def __init__(self,dimension: int = 384):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[EmbeddedChunk] = []

    def add(
            self,
            embedded_chunks: list[EmbeddedChunk]
    )-> None:
        if not embedded_chunks:
            return

        embeddings = np.array(
            [item.embedding for item in embedded_chunks],
            dtype="float32"
        )

        self.index.add(embeddings)
        self.chunks.extend(embedded_chunks)

    def search(
        self,
        query_embedding,
        top_k: int = 3
    ) -> list[tuple[EmbeddedChunk, float]]:
        query = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (self.chunks[index], float(score))
            )

        return results