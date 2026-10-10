from src.citation_validator import validate_citation_numbers


def test_valid_citations():
    source_mapping = {1: {}, 2: {}, 3: {}}

    answer = "CO2e is carbon dioxide equivalent. [Source 3]"

    assert validate_citation_numbers(
        answer,
        source_mapping,
    ) == []


def test_invalid_citation():
    source_mapping = {1: {}, 2: {}, 3: {}}

    answer = "CO2e is carbon dioxide equivalent. [Source 7]"

    assert validate_citation_numbers(
        answer,
        source_mapping,
    ) == [7]


def test_mixed_citations():
    source_mapping = {1: {}, 2: {}, 3: {}}

    answer = "CO2e is a common unit. [Source 1] More details [Source 7]."

    assert validate_citation_numbers(
        answer,
        source_mapping,
    ) == [7]