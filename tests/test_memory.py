from src.memory import ConversationMemory


def test_add_and_retrieve_turn():
    memory = ConversationMemory()

    memory.add_turn(
        "What does CO2e mean?",
        "CO2e means carbon dioxide equivalent.",
    )

    history = memory.get_history()

    assert len(history) == 1
    assert history[0]["question"] == "What does CO2e mean?"


def test_memory_keeps_only_recent_turns():
    memory = ConversationMemory(max_turns=2)

    memory.add_turn("Question 1", "Answer 1")
    memory.add_turn("Question 2", "Answer 2")
    memory.add_turn("Question 3", "Answer 3")

    history = memory.get_history()

    assert len(history) == 2
    assert history[0]["question"] == "Question 2"
    assert history[1]["question"] == "Question 3"


def test_clear_memory():
    memory = ConversationMemory()

    memory.add_turn("Question", "Answer")
    memory.clear()

    assert memory.get_history() == []
