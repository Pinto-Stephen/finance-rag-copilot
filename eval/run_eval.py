"""
run_eval.py — Phase 6: evaluate the RAG pipeline with RAGAS.

For each question in eval/testset.json it runs the REAL pipeline (retrieve +
rerank + generate), then scores four metrics that deliberately separate
retrieval quality from generation quality:

  - Faithfulness                        : is the answer grounded in the retrieved
                                          chunks? (catches hallucination)
  - ResponseRelevancy                   : does the answer address the question?
                                          (uses embeddings)
  - LLMContextPrecisionWithoutReference : are the retrieved chunks actually relevant?
  - LLMContextRecall                    : did retrieval surface what the reference
                                          answer needs? (uses the reference)

The judge LLM and embeddings are YOUR Groq + BGE-M3 — RAGAS would otherwise
default to OpenAI and demand an OpenAI key.

Run from project root:
    python -m eval.run_eval
"""

import json
import sys
import types
from pathlib import Path

# ragas 0.4.x hard-imports `langchain_community.chat_models.vertexai.ChatVertexAI`
# at package init, but langchain-community 0.4.x (required by the langchain 1.x
# stack langgraph/langchain-groq depend on) removed that module. Our judge is Groq,
# so Vertex AI is never used — register a harmless stub so the import resolves.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from src.retrieve import load_index
from src.generate import answer, MODEL

load_dotenv()

TESTSET = Path("eval/testset.json")
RESULTS = Path("eval/results.csv")


def build_samples(index):
    """Run the pipeline on each test question and wrap results for RAGAS."""
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))
    samples = []
    for case in cases:
        response, nodes = answer(
            case["question"],
            index,
            company=case.get("company"),
            year=case.get("year"),
        )
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                retrieved_contexts=[n.text for n in nodes],
                response=response,
                reference=case.get("reference", ""),
            )
        )
        print(f"  ran: {case['question'][:60]}...")
    return samples


def main():
    index = load_index()

    print("Running the pipeline over the test set...")
    dataset = EvaluationDataset(samples=build_samples(index))

    # Judge LLM + embeddings = your own Groq + BGE-M3.
    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    print("Scoring with RAGAS (several judge-LLM calls per sample)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    # Per-sample breakdown — this is where you diagnose which questions failed
    # and whether the failure was retrieval (low context recall/precision) or
    # generation (low faithfulness).
    result.to_pandas().to_csv(RESULTS, index=False)
    print(f"\nPer-sample results written to {RESULTS}")


if __name__ == "__main__":
    main()