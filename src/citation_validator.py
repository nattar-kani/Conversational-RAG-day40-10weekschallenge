import re


def validate_citation_numbers(answer, source_mapping):
    cited_numbers = re.findall(
        r"\[Source (\d+)\]",
        answer,
    )

    invalid_citations = []

    for number in cited_numbers:
        citation_number = int(number)

        if citation_number not in source_mapping:
            invalid_citations.append(citation_number)

    return invalid_citations

