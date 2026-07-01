import os

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq

from src.retrieve import load_index, retrieve
from config.settings import LLM_MODEL, DEFAULT_CORPUS, CORPORA

load_dotenv()

# Single source of truth lives in config.settings; re-exported as MODEL so
# agent.py and eval/run_eval.py keep importing it from here.
MODEL = LLM_MODEL

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set — add it to your .env file.")

# temperature=0 keeps answers deterministic and minimizes hallucination for a fact-extraction task.
llm = Groq(model=MODEL, api_key=GROQ_API_KEY, temperature=0)

# Per-corpus prompt framing. Only the human-readable wording changes per corpus — the
# citation *mechanism* (format_context's bracketed tags) is untouched; `tag` here just
# mirrors what those tags look like so the model echoes them. SEC's values reproduce the
# original prompt BYTE-FOR-BYTE (backward-compat is a hard requirement); NASA/RBI get
# corpus-appropriate framing. Keyed by the same corpus keys as config.settings.CORPORA.
_FRAMING = {
    "sec": {
        "assistant": "financial research assistant",
        "docs": "US airline 10-K filings",
        "docs_short": "10-K filings",
        "source": "source filing",
        "tag": "[TICKER YEAR]",
    },
    "nasa": {
        "assistant": "research assistant",
        "docs": "NASA technical reports",
        "docs_short": "NASA technical reports",
        "source": "source report",
        "tag": "[NASA TR ...]",
    },
    "rbi": {
        "assistant": "research assistant",
        "docs": "RBI circulars",
        "docs_short": "RBI circulars",
        "source": "source circular",
        "tag": "[RBI — ...]",
    },
}
assert set(_FRAMING) == set(CORPORA), "prompt framing must cover exactly the CORPORA keys"


def _system_prompt(corpus=DEFAULT_CORPUS):
    """Build the system prompt for `corpus`. SEC's output is byte-identical to the
    original hardcoded prompt; other corpora swap in their own framing."""
    f = _FRAMING[corpus]
    return (
        f"You are a {f['assistant']} answering questions about {f['docs']}. "
        "Follow these rules strictly:\n"
        "1. Answer ONLY using the provided context. Never use outside knowledge.\n"
        f"2. Cite the {f['source']} for every claim using its {f['tag']} tag.\n"
        "3. If the context does not contain the answer, say you don't know — do not guess.\n"
        "4. Be concise and factual."
    )


def _prompt_template(corpus=DEFAULT_CORPUS):
    """Build the user-prompt template for `corpus` (keeps {context}/{question}
    placeholders for the later .format()). SEC's output is byte-identical."""
    f = _FRAMING[corpus]
    return (
        f"Context from {f['docs_short']}:\n"
        "========================\n"
        "{context}\n"
        "========================\n\n"
        "Question: {question}\n\n"
        f"Answer (cite each claim as {f['tag']}):"
    )


# Module-level SEC prompts, kept for back-compat / byte-identity checks.
SYSTEM_PROMPT = _system_prompt(DEFAULT_CORPUS)
PROMPT_TEMPLATE = _prompt_template(DEFAULT_CORPUS)


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

    prompt = _prompt_template(corpus).format(
        context=format_context(nodes, corpus), question=question
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=_system_prompt(corpus)),
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
