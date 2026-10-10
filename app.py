from pathlib import Path
import streamlit as st

from src.citation_validator import validate_citation_numbers
from src.memory import ConversationMemory
from src.streaming_rag import prepare_answer, stream_answer


st.set_page_config(
    page_title="GroundTruth",
    page_icon="G",
    layout="centered",
)

st.title("GroundTruth")
st.caption(
    "A conversational RAG assistant with hybrid retrieval, source citations, conversation memory, and an abstention guardrail for questions that lack sufficient evidence."
)

st.info(
    "Ask questions about the documents in the knowledge base. "
    "Answers are generated from retrieved context."
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if not sources:
        return

    with st.expander(f"View retrieved sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f"**[Source {source['number']}] "
                f"{source['source']}**"
            )
            st.caption(
                f"Chunk {source['chunk_index']} · "
                f"RRF score: {source['rrf_score']:.4f}"
            )
            st.write(source["content"])
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

            if message.get("invalid_citations"):
                st.warning(
                    "Some citation numbers do not match "
                    "the retrieved sources."
                )


question = st.chat_input("Ask a question about the documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant context..."):
                prepared = prepare_answer(
                    question,
                    st.session_state.memory,
                )

            sources = prepared["sources"]

            if prepared["fallback"]:
                answer = prepared["answer"]
                st.markdown(answer)

                st.session_state.memory.add_turn(
                    question,
                    answer,
                )
                invalid_citations = []

            else:
                st.caption(
                    f"Best dense similarity: "
                    f"{prepared['best_similarity']:.3f}"
                )

                answer = st.write_stream(
                    stream_answer(
                        prepared,
                        question,
                        st.session_state.memory,
                    )
                )

                source_mapping = {
                    source["number"]: source
                    for source in sources
                }

                invalid_citations = validate_citation_numbers(
                    answer,
                    source_mapping,
                )

                if invalid_citations:
                    st.warning(
                        "Invalid citation numbers detected: "
                        f"{invalid_citations}"
                    )

            show_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "invalid_citations": invalid_citations,
                }
            )

        except Exception:
            st.error(
                "GroundTruth could not generate an answer. "
                "Check that Ollama is running and the model "
                "is available."
            )
            st.exception(Exception)


if st.sidebar.button("Clear conversation"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("### About")
st.sidebar.caption(
    "Answers are generated from the documents in this knowledge base."
)

st.sidebar.write("GroundTruth")
st.sidebar.write("Local inference with Ollama")
st.sidebar.caption(
    "Similarity is a retrieval signal, not a guarantee "
    "that an answer is correct."
    "Built as part of 10 weeks AI Engineer Challenge"
)

st.sidebar.divider()

st.sidebar.markdown("### Knowledge Base")

document_path = Path(__file__).parent / "data" / "raw" / "sample.txt"

if document_path.exists():
    document_content = document_path.read_text(encoding="utf-8")

    with st.sidebar.expander("📄 sample.txt — View content"):
        st.text(document_content)

else:
    st.sidebar.warning("sample.txt was not found.")




