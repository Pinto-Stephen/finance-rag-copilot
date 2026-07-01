import os

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq

from src.retrieve import load_index, retrieve
from config.settings import LLM_MODEL, DEFAULT_CORPUS

load_dotenv()

# Single source of truth lives in config.settings; re-exported as MODEL so
# agent.py and eval/run_eval.py keep importing it from here.
MODEL = LLM_MODEL

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set — add it to your .env file.")

# temperature=0 keeps answers deterministic and minimizes hallucination for a fact-extraction task.
llm = Groq(model=MODEL, api_key=GROQ_API_KEY, temperature=0)

SYSTEM_PROMPT = (
    "You are a financial research assistant answering questions about US airline "
    "10-K filings. Follow these rules strictly:\n"
    "1. Answer ONLY using the provided context. Never use outside knowledge.\n"
    "2. Cite the source filing for every claim using its [TICKER YEAR] tag.\n"
    "3. If the context does not contain the answer, say you don't know — do not guess.\n"
    "4. Be concise and factual."
)

PROMPT_TEMPLATE = (
    "Context from 10-K filings:\n"
    "========================\n"
    "{context}\n"
    "========================\n\n"
    "Question: {question}\n\n"
    "Answer (cite each claim as [TICKER YEAR]):"
)


def format_context(nodes, corpus=DEFAULT_CORPUS):
    """Tag each chunk with its source so the model can cite it. SEC keeps its
    [TICKER YEAR] tag (unchanged); other corpora tag with their citation/title,
    since they carry no company/year in the uniform metadata schema."""
    blocks = []
    for node in nodes:
        m = node.metadata
        if corpus == DEFAULT_CORPUS:
            tag = f"[{m.get('company')} {m.get('year')}]"
        else:
            tag = f"[{m.get('citation') or m.get('title') or m.get('doc_id')}]"
        blocks.append(f"{tag}\n{node.text}")
    return "\n\n".join(blocks)


def answer(question, index, corpus=DEFAULT_CORPUS, company=None, year=None):
    """Retrieve + rerank, then generate a grounded, cited answer.

    Returns (answer_text, source_nodes) so the app/other callers can
    show the answer alongside the filings it came from"""
    nodes = retrieve(question, index, corpus=corpus, company=company, year=year)
    if not nodes:
        return "No relevant filings found for that scope.", []

    prompt = PROMPT_TEMPLATE.format(context=format_context(nodes, corpus), question=question)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    response = llm.chat(messages)
    return response.message.content, nodes


if __name__ == "__main__":
    index = load_index()

    q = "What did Delta say about fuel hedging?"
    text, sources = answer(q, index, company="DAL")

    print(f"Q: {q}\n")
    print(text)
    print("\n--- sources used ---")
    for node in sources:
        m = node.metadata
        print(f"[{m.get('company')} {m.get('year')}]  score={node.score:.3f}")
