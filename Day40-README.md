# GroundTruth: Conversational RAG with Citations and Guardrails

**Day 40 | 70 Days / 10 Week AI Engineer Challenge**

GroundTruth is a local conversational Retrieval-Augmented Generation (RAG) application designed to answer questions using retrieved document context. It combines hybrid retrieval, query rewriting, source attribution, short-term conversational memory, and a similarity-based guardrail to reduce unsupported answers.

The goal is not just to generate an answer, but to make it easier to trace answers back to source material and recognise when the available context may be insufficient.

## Features

- **Hybrid retrieval:** Combines dense vector search with BM25 keyword search.
- **Reciprocal Rank Fusion (RRF):** Combines rankings from both retrieval methods.
- **Conversational memory:** Keeps the most recent five question-and-answer turns.
- **Query rewriting:** Converts follow-up questions into standalone retrieval queries using conversation history.
- **Source attribution:** Labels retrieved chunks as `[Source 1]`, `[Source 2]`, and so on, and asks the LLM to cite them in its response.
- **Citation validation:** Checks whether citation numbers used in an answer correspond to retrieved sources.
- **Abstention guardrail:** Returns a fallback response when the best dense similarity score falls below a configurable threshold.
- **Local inference:** Uses Ollama to generate answers without relying on a hosted LLM API.

## Architecture

```text
User Question
     |
     v
Conversation Memory
     |
     v
Query Rewriting (Ollama)
     |
     v
Hybrid Retrieval
  /           \
BM25       Dense Search
  \           /
     RRF
      |
      v
Retrieved Source Chunks
      |
      v
Similarity Guardrail
      |
      +---- Insufficient similarity
      |             |
      |             v
      |       Fallback Answer
      |
      v
Context + Citation Instructions
      |
      v
Ollama Answer Generation
      |
      v
Citation Number Validation
      |
      v
Save Turn to Memory
      |
      v
Display Answer
```

## Tech Stack

- Python
- Ollama
- Qwen 2.5 1.5B (`qwen2.5:1.5b`)
- Sentence Transformers
- `all-MiniLM-L6-v2` embeddings
- FAISS for dense vector search
- BM25 using `rank_bm25`
- Reciprocal Rank Fusion for hybrid ranking
- pytest for unit testing

## How It Works

### 1. Hybrid Retrieval

Dense retrieval finds semantically similar chunks using sentence embeddings and FAISS. BM25 retrieves chunks based on keyword overlap.

The two ranked lists are combined using Reciprocal Rank Fusion (RRF):

\[
RRF(d) = \sum_{r \in R(d)} \frac{1}{k + r}
\]

Here, `r` is a document's rank in a result list and `k` is a ranking constant set to 60.

RRF combines rankings without requiring the raw scores from the two retrieval systems to be directly comparable.

**Important:** An RRF score is a ranking signal, not a probability that an answer is correct.

### 2. Conversational Memory and Query Rewriting

Follow-up questions often depend on earlier messages.

For example:

- First question: *What is CO2e?*
- Follow-up: *How is it measured?*

The second question may not contain enough information for effective retrieval on its own. GroundTruth uses recent conversation history to rewrite it into a standalone query before searching the document collection.

The original question and generated answer are stored in memory. Only the most recent five turns are retained during the current session.

### 3. Source Attribution

Retrieved chunks are assigned temporary source numbers and passed to the LLM with their content.

The model is instructed to answer using the supplied context and include citations such as `[Source 1]`.

A citation mapping connects each source number to its filename, chunk index, content, and retrieval score.

The current citation validator checks whether citation numbers refer to sources that were actually retrieved. It does not independently verify that every factual claim is supported by its cited passage.

### 4. Honest Abstention

GroundTruth compares the best dense similarity score against a configurable threshold:

```python
FALLBACK_ANSWER = (
    "I don't have enough information in the provided context."
)

MIN_SIMILARITY = 0.35
```

If the best score is below the threshold, the application returns the fallback answer rather than asking the LLM to generate a response from weakly matching context.

This is a simple baseline guardrail, not a guarantee against hallucinations. Similarity thresholds should be calibrated against representative in-scope and out-of-scope questions.

## Setup and Usage

### Prerequisites

- Python and the project's existing Python environment
- Ollama installed and running locally
- The `qwen2.5:1.5b` model available in Ollama
- Project dependencies installed

If the model is not available locally, download it with:

```powershell
ollama pull qwen2.5:1.5b
```

### Run GroundTruth

Activate the Python environment used for the project, navigate to the project root, and run:

```powershell
python -m src.ollama_rag
```

Ask questions in the interactive terminal. Type `exit` or `quit` to end the conversation.

The embedding model `all-MiniLM-L6-v2` may be downloaded automatically the first time it is loaded.

### Run Tests

Run the unit tests:

```powershell
python -m pytest tests/test_memory.py tests/test_citation_validator.py -v
```

Or run all discovered tests:

```powershell
python -m pytest -v
```

## Validation and Observed Results

The following checks were performed during development.

| Test scenario | Observed result |
|---|---|
| Ask about CO2e | Retrieved relevant document chunks and generated an answer |
| Ask the follow-up, “How is it measured?” | Rewrote the query using conversation context and retrieved relevant chunks |
| Ask about the capital of India | Returned the fallback answer because the best dense similarity score was 0.1211, below the 0.35 threshold |
| Exit the interactive session | Printed `Conversation ended.` and returned to the terminal |
| Conversation memory tests | 3 tests passed |
| Citation validator tests | 3 tests passed |

The follow-up question was rewritten into a standalone query about measuring carbon dioxide equivalent in the context of comparing greenhouse gas impacts. This confirmed that query rewriting was functioning as intended.

The out-of-scope question demonstrated the similarity guardrail's fallback behaviour.

These results are observations from the current sample dataset and test run, not a comprehensive evaluation of answer accuracy or citation quality.






