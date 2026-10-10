
import re


def validate_citation_numbers(answer, source_mapping):
    cited_numbers = re.findall(r"\[Source (\d+)\]", answer)

    invalid_citations = [
        int(number)
        for number in cited_numbers
        if int(number) not in source_mapping
    ]

    return invalid_citations
