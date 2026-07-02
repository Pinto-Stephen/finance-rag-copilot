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

import argparse
import ast
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
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from src.retrieve import load_index
from src.generate import answer, MODEL
from config.settings import DEFAULT_CORPUS

load_dotenv()

TESTSET = Path("eval/testset.json")
RESULTS = Path("eval/results.csv")

# The RAG pipeline answers with gpt-oss-120b (MODEL). The RAGAS judge, however, fans
# out several LLM calls per sample across four metrics (LLMContextPrecision alone judges
# each retrieved chunk), so pointing it at gpt-oss exhausts that model's per-day token
# budget and starves scoring. Judge on a separate Groq model with the largest free-tier
# daily budget (llama-3.1-8b-instant, ~500k tokens/day) so judging never competes with
# generation and a full 4-metric run over the whole testset fits in one day.
JUDGE_MODEL = "llama-3.1-8b-instant"


def build_samples():
    """Run the pipeline on each test question and wrap results for RAGAS. Each case
    is routed to its own corpus (defaults to SEC when omitted); indexes are loaded
    once per corpus and reused."""
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))
    indexes = {}
    samples, corpora = [], []
    for case in cases:
        corpus = case.get("corpus", DEFAULT_CORPUS)
        if corpus not in indexes:
            indexes[corpus] = load_index(corpus)
        response, nodes = answer(
            case["question"],
            indexes[corpus],
            corpus=corpus,
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
        corpora.append(corpus)
        print(f"  ran [{corpus}]: {case['question'][:55]}...")
    return samples, corpora


def load_cached_samples(limit_per_corpus=None):
    """Rebuild RAGAS samples from a previous run's results.csv — the genuine pipeline
    responses and retrieved contexts — so scoring can be re-run WITHOUT calling the
    generation model again. Useful when the answer model's daily token quota is
    exhausted but the judge model still has budget.

    limit_per_corpus keeps only the first N rows of each corpus, so a full 4-metric
    scoring pass fits a fast judge's daily token budget and finishes in one sitting."""
    import pandas as pd

    df = pd.read_csv(RESULTS)
    if limit_per_corpus:
        df = df.groupby("corpus", sort=False).head(limit_per_corpus).reset_index(drop=True)
    samples, corpora = [], []
    for _, r in df.iterrows():
        contexts = ast.literal_eval(r["retrieved_contexts"]) if isinstance(
            r["retrieved_contexts"], str) else []
        samples.append(
            SingleTurnSample(
                user_input=r["user_input"],
                retrieved_contexts=contexts,
                response=r["response"],
                reference=r["reference"] if pd.notna(r["reference"]) else "",
            )
        )
        corpora.append(r["corpus"])
    print(f"Loaded {len(samples)} cached responses from {RESULTS} (no generation).")
    return samples, corpora


def score_and_report(samples, corpora, judge_model=JUDGE_MODEL, max_workers=5):
    """Score samples with the RAGAS judge and write/print the per-corpus breakdown."""
    dataset = EvaluationDataset(samples=samples)

    # Judge LLM + embeddings = your own BGE-M3. bypass_n=True makes RAGAS issue n separate
    # single-completion calls instead of sending n>1 in one request, which Groq rejects
    # (HTTP 400: "'n' : number must be at most 1"). max_tokens is generous so multi-step
    # judgments (claim NLI, per-chunk verdicts) don't get truncated (LLMDidNotFinish).
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(model=judge_model, temperature=0, max_tokens=2048), bypass_n=True
    )
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )

    # strictness=1: ResponseRelevancy generates `strictness` reverse-questions per sample;
    # 1 keeps the judge token budget within the free-tier daily cap (default is 3).
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    # The 8B judge is slow (~30-60s/call at 6k TPM) and some metrics chain several calls
    # per sample (context-precision judges each retrieved chunk), so a low timeout kills
    # those jobs mid-flight -> NaN. Keep concurrency modest (less per-minute contention,
    # so each job's sequential calls run without throttling) and the timeout generous.
    run_config = RunConfig(max_workers=max_workers, timeout=600, max_retries=4)

    print(f"Scoring with RAGAS (judge={judge_model}, several calls per sample)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
    )

    print("\n=== Aggregate scores ===")
    print(result)

    # Per-sample breakdown — this is where you diagnose which questions failed
    # and whether the failure was retrieval (low context recall/precision) or
    # generation (low faithfulness). A `corpus` column is added so results can be
    # sliced per corpus.
    df = result.to_pandas()
    df.insert(0, "corpus", corpora)
    df.to_csv(RESULTS, index=False)
    print(f"\nPer-sample results written to {RESULTS}")

    metric_cols = [c for c in df.columns if c not in
                   ("corpus", "user_input", "retrieved_contexts", "response", "reference")]
    print("\n=== Per-corpus means ===")
    print(df.groupby("corpus")[metric_cols].mean().round(3).to_string())
    print("\n=== Overall means ===")
    print(df[metric_cols].mean().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score the responses already in results.csv instead of regenerating "
             "them (skips the generation model — use when its daily quota is exhausted).",
    )
    parser.add_argument(
        "--judge", default=JUDGE_MODEL,
        help=f"Groq model to use as the RAGAS judge (default: {JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--limit-per-corpus", type=int, default=None,
        help="Score only the first N cached samples per corpus (used with --rescore) so "
             "a full 4-metric pass fits a fast judge's daily token budget.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="RAGAS scoring concurrency (default 5).",
    )
    args = parser.parse_args()

    if args.rescore:
        samples, corpora = load_cached_samples(limit_per_corpus=args.limit_per_corpus)
    else:
        print("Running the pipeline over the test set...")
        samples, corpora = build_samples()

    score_and_report(samples, corpora, judge_model=args.judge, max_workers=args.max_workers)


if __name__ == "__main__":
    main()