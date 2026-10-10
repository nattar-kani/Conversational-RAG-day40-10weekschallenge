from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    content: str
    source: str
    file_type: str
    metadata: dict[str,Any] = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: Any