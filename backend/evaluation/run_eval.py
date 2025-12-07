"""
RAGAS Evaluation Runner for micro-rag.

This script:
1. Loads the evaluation dataset
2. Runs each question through the RAG pipeline
3. Evaluates using RAGAS metrics
4. Outputs results

Usage:
    cd backend
    PYTHONPATH=. python evaluation/run_eval.py
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from sqlalchemy import select

import app.models.database as database
from app.models.database import init_db, get_db, Collection
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore


EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


async def run_rag_for_eval(questions: list[str], collection_id: int) -> list[dict]:
    """
    Run RAG pipeline for each question and collect results.

    Returns list of dicts with: question, answer, contexts, ground_truth
    """
    results = []

    async for db in get_db():
        engine = RAGEngine(db)
        vs = VectorStore(db)

        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q[:50]}...")

            # Get retrieved chunks
            chunks = await vs.search(
                query=q,
                collection_id=collection_id,
                limit=10,
                score_threshold=0.4,
            )

            # Get RAG answer
            response = await engine.query(
                question=q,
                collection_id=collection_id,
                top_k=10,
            )

            results.append({
                "question": q,
                "answer": response["answer"],
                "contexts": [c["content"] for c in chunks],
            })

        break  # Only need one db session

    return results


def load_eval_dataset() -> list[dict]:
    """Load evaluation dataset from JSON file."""
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


async def main():
    print("=" * 60)
    print("RAGAS Evaluation Runner")
    print("=" * 60)

    # Initialize database
    print("\n1. Initializing database...")
    await init_db()

    # Get collection
    async for db in get_db():
        result = await db.execute(
            select(Collection).where(Collection.slug == "eu5-wiki")
        )
        collection = result.scalar_one_or_none()
        if not collection:
            print("ERROR: Collection 'eu5-wiki' not found!")
            return
        print(f"   Collection: {collection.name} (ID: {collection.id})")
        break

    # Load eval dataset
    print("\n2. Loading evaluation dataset...")
    eval_data = load_eval_dataset()
    print(f"   Loaded {len(eval_data)} test questions")

    # Run RAG for each question
    print("\n3. Running RAG pipeline...")
    questions = [d["question"] for d in eval_data]
    ground_truths = [d["ground_truth"] for d in eval_data]

    rag_results = await run_rag_for_eval(questions, collection.id)

    # Combine with ground truth
    for i, result in enumerate(rag_results):
        result["ground_truth"] = ground_truths[i]

    # Prepare dataset for RAGAS
    print("\n4. Preparing RAGAS dataset...")
    ragas_data = {
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "ground_truth": [r["ground_truth"] for r in rag_results],
    }
    dataset = Dataset.from_dict(ragas_data)

    # Run RAGAS evaluation
    print("\n5. Running RAGAS evaluation...")
    print("   (This may take a few minutes...)")

    # Configure LLM for RAGAS (uses OPENAI_API_KEY from environment)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    metrics = [
        faithfulness,
        answer_relevancy,
    ]

    results = evaluate(dataset, metrics=metrics, llm=evaluator_llm)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Convert to pandas for easier processing
    results_df = results.to_pandas()

    # Calculate overall scores (mean of each metric)
    print("\nOverall Scores:")
    metric_columns = ['faithfulness', 'answer_relevancy']
    overall_scores = {}
    for col in metric_columns:
        if col in results_df.columns:
            score = results_df[col].mean()
            overall_scores[col] = score
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {col:25} {bar} {score:.3f}")

    # Save detailed results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"eval_{timestamp}.json"

    # Add per-question scores to rag_results
    for i, r in enumerate(rag_results):
        for col in metric_columns:
            if col in results_df.columns:
                r[col] = float(results_df.iloc[i][col]) if not pd.isna(results_df.iloc[i][col]) else None

    # Convert results to serializable format
    output = {
        "timestamp": timestamp,
        "num_questions": len(eval_data),
        "overall_scores": overall_scores,
        "per_question": rag_results,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    # Print per-question summary
    print("\nPer-Question Summary:")
    print("-" * 60)
    for i, r in enumerate(rag_results):
        q_short = r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"]
        faith = r.get("faithfulness", "N/A")
        relevancy = r.get("answer_relevancy", "N/A")
        faith_str = f"{faith:.2f}" if isinstance(faith, float) else faith
        rel_str = f"{relevancy:.2f}" if isinstance(relevancy, float) else relevancy
        print(f"Q{i+1}: {q_short}")
        print(f"    Faith: {faith_str} | Relevancy: {rel_str}")


if __name__ == "__main__":
    asyncio.run(main())
