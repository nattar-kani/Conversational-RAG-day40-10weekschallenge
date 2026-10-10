
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

from src.ragas_embeddings import ragas_embeddings


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ragas_dataset"
)
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_NAME = "qwen2.5:1.5b"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_from_disk(str(DATASET_PATH))

    required_columns = {
        "question",
        "answer",
        "contexts",
        "ground_truth",
    }
    missing = required_columns - set(dataset.column_names)

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    print(f"Loaded {len(dataset)} saved answers.")

    evaluator_llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    print(f"Evaluator: local Ollama / {MODEL_NAME}")
    print("Running RAGAS metrics...")

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=ragas_embeddings,
    )

    scores = dict(result)

    json_path = OUTPUT_DIR / "ragas_metrics.json"
    csv_path = OUTPUT_DIR / "ragas_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": str(DATASET_PATH),
                "dataset_rows": len(dataset),
                "evaluator_model": MODEL_NAME,
                "metrics": {
                    key: (
                        float(value)
                        if value is not None
                        else None
                    )
                    for key, value in scores.items()
                },
            },
            file,
            indent=2,
        )

    pd.DataFrame(
        [
            {"metric": key, "score": value}
            for key, value in scores.items()
        ]
    ).to_csv(csv_path, index=False)

    print("\nEvaluation results:")
    for metric, score in scores.items():
        print(f"{metric}: {score}")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
