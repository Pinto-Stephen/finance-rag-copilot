import os

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq

from src.retrieve import load_index, retrieve

load_dotenv()

MODEL = "openai/gpt-oss-120b"

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


def format_context(nodes):
    """Tag each chunk with its source filing so the model can cite it."""
    blocks = []
    for node in nodes:
        m = node.metadata
        tag = f"[{m.get('company')} {m.get('year')}]"
        blocks.append(f"{tag}\n{node.text}")
    return "\n\n".join(blocks)


def answer(question, index, company=None, year=None):
    """Retrieve + rerank, then generate a grounded, cited answer.

    Returns (answer_text, source_nodes) so the app/other callers can
    show the answer alongside the filings it came from"""
    nodes = retrieve(question, index, company=company, year=year)
    if not nodes:
        return "No relevant filings found for that scope.", []

    prompt = PROMPT_TEMPLATE.format(context=format_context(nodes), question=question)
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
