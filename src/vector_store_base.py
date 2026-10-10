from abc import ABC, abstractmethod

from .models import EmbeddedChunk

class BaseVectorStore(ABC):

    @abstractmethod
    def add(
        self,
        embedded_chunk: list[EmbeddedChunk]
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding,
        top_k: int = 3
    ) -> list[tuple[EmbeddedChunk,float]]:
        pass