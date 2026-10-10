import re 

def build_citation_context(results):
    context_parts = []
    source_mapping = {}

    for citation_number, (chunk,rrf_score) in enumerate(
        results,
        start=1
    ):
        
        source_mapping[citation_number] = {
            "source": chunk.source,
            "chunk_index":chunk.chunk_index,
            "content": chunk.content,
            "rrf_score": rrf_score
        }

        context_parts.append(
            f"""[Source {citation_number}]
            File: {chunk.source}
            Chunk: {chunk.chunk_index}
            Content: {chunk.content}"""
        )

        context = "\n\n".join(context_parts)

        return context, source_mapping

def validate_citation_numbers(answer,source_mapping):
    cited_numbers = re.findall(
        r"\[Source (\d+)\]",
        answer
    )

    invalid_citations = []

    for number in cited_numbers:
        citation_number = int(number)

        if citation_number not in source_mapping:
            invalid_citations.append(citation_number)

        return invalid_citations



























