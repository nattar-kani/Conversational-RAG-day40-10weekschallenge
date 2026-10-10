from sentence_transformers import SentenceTransformer
from ragas.embeddings.base import BaseRagasEmbeddings


class LocalRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


ragas_embeddings = LocalRagasEmbeddings()