from sentence_transformers import SentenceTransformer

from .models import Chunk, EmbeddedChunk


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [chunk.content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    def embed_query(self, query: str):
        
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )