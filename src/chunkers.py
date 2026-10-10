from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Document, Chunk

def fixed_size_chunks(
        document: Document,
        chunk_size: int=500
) -> list[Chunk]:
    content = document.content

    chunks = []

    for index,start in enumerate(range(0,len(content),chunk_size)):
        chunk = content[start:start+chunk_size]

        if chunk.strip():
            chunks.append(
                Chunk(
                    content=chunk,
                    source=document.source,
                    chunk_index=index,
                    strategy="fixed_size",
                    metadata=document.metadata.copy(),
                )
            )

    return chunks

def recursive_chunks(
    document: Document,
    chunk_size: int=500,
    chunk_overlap: int=50
)-> list[Chunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    texts = splitter.split_text(document.content)

    chunks = []

    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                content=text,
                source=document.source,
                chunk_index=index,
                strategy="recursive",
                metadata=document.metadata.copy(),
            )
        )

    return chunks

def sentence_chunks(
    document: Document,
    chunk_size: int = 500
) -> list[Chunk]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        document.content.strip()
    )

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if (
            current_sentences
            and current_length + len(sentence) > chunk_size
        ):
            chunks.append(
                Chunk(
                    content=" ".join(current_sentences),
                    source=document.source,
                    chunk_index=len(chunks),
                    strategy="sentence",
                    metadata=document.metadata.copy()
                )
            )

            current_sentences = []
            current_length = 0

        current_sentences.append(sentence)
        current_length += len(sentence) + 1

    if current_sentences:
        chunks.append(
            Chunk(
                content=" ".join(current_sentences),
                source=document.source,
                chunk_index=len(chunks),
                strategy="sentence",
                metadata=document.metadata.copy()
            )
        )

    return chunks

def semantic_chunks(
    document: Document,
    similarity_threshold: float = 0.5,
    model=None
) -> list[Chunk]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        document.content.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return []

    if len(sentences) == 1:
        return [
            Chunk(
                content=sentences[0],
                source=document.source,
                chunk_index=0,
                strategy="semantic",
                metadata=document.metadata.copy(),
            )
        ]
    
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
    )

    chunks = []
    current_sentences = [sentences[0]]

    for index in range(len(sentences) - 1):

        similarity = np.dot(
            embeddings[index],
            embeddings[index + 1],
        )

        if similarity < similarity_threshold:

            chunks.append(
                Chunk(
                    content=" ".join(current_sentences),
                    source=document.source,
                    chunk_index=len(chunks),
                    strategy="semantic",
                    metadata=document.metadata.copy(),
                )
            )

            current_sentences = []

        current_sentences.append(sentences[index + 1])

    if current_sentences:
        chunks.append(
            Chunk(
                content=" ".join(current_sentences),
                source=document.source,
                chunk_index=len(chunks),
                strategy="semantic",
                metadata=document.metadata.copy(),
            )
        )

    return chunks