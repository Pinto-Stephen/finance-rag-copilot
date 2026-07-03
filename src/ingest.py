"""
ingest.py — Load a corpus (SEC 10-K HTML, or NASA/RBI PDFs), clean its text,
attach a uniform metadata schema, and split into chunks (nodes) ready for embedding.

Run directly from the project root to sanity-check the load/chunk step:
    python -m src.ingest

Import it from index_build.py to get the nodes without side effects:
    from src.ingest import load_and_chunk             # SEC default (back-compat)
    from src.ingest import load_and_chunk_corpus      # any corpus: sec | nasa | rbi
    docs, nodes = load_and_chunk_corpus("nasa")

Each corpus uses its own reader (HTML/XBRL for SEC, PDF for NASA/RBI) but they all
emit the SAME metadata keys — corpus/title/citation/source_url/doc_id — so retrieval,
citations and the UI can treat the three collections uniformly downstream.
"""

import json
import logging
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_DATA_DIR, CORPORA

logger = logging.getLogger(__name__)

# 10-Ks are XBRL parsed as HTML, so this warning would otherwise fire constantly.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:                          # pragma: no cover - environment-dependent
    fitz = None
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except ImportError:
        pypdf = None
        PDF_BACKEND = None


_LARGE_PDF_MB = 25
_MIN_TEXT_CHARS = 200

_NON_CONTEXT_KEYS = ["source_url", "doc_id", "file_path", "file_name"]



# 10-Ks are inline-XBRL: a big block of machine-readable facts (contexts, units,
# dimension members like "us-gaap:FairValueMeasurementsRecurringMember", CIKs and
# period dates) is embedded in the page header. get_text() would otherwise pull all
# of that tag-soup into the corpus and bloat retrieval. We drop those metadata
# blocks but keep the inline <ix:nonFraction>/<ix:nonNumeric> wrappers, because
# those hold the *visible* reported numbers and narrative.
_XBRL_NOISE_TAGS = {
    "script", "style", "head",
    "ix:header", "ix:hidden", "ix:references", "ix:resources",
}


def _make_document(text, metadata):
    """Wrap cleaned text + metadata in a Document, marking provenance keys
    (source_url/doc_id/file_*) as excluded from both the embedding and the LLM
    context. These exclusions propagate to every node the splitter derives."""
    doc = Document(text=text, metadata=metadata or {})
    doc.excluded_embed_metadata_keys = list(_NON_CONTEXT_KEYS)
    doc.excluded_llm_metadata_keys = list(_NON_CONTEXT_KEYS)
    return doc


class HTMLTextReader(BaseReader):
    """Strip a 10-K's HTML/inline-XBRL down to readable narrative text."""

    def load_data(self, file, extra_info=None):
        html = Path(file).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(
            lambda t: t.name and t.name.lower() in _XBRL_NOISE_TAGS
        ):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
        return [_make_document(text, extra_info or {})]


def _extract_pdf_text(path):
    """Extract a PDF's full text via the available backend (PyMuPDF, else pypdf)."""
    if fitz is not None:
        with fitz.open(path) as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if pypdf is not None:
        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise RuntimeError(
        "No PDF backend available — install pymupdf (preferred) or pypdf."
    )


class PDFTextReader(BaseReader):
    """Extract a NASA/RBI PDF down to clean narrative text — the PDF analogue of
    HTMLTextReader. Logs a heads-up for very large PDFs and skips (returns no
    Document) any PDF whose text layer is too thin to be a real document."""

    def load_data(self, file, extra_info=None):
        path = Path(file)
        size_mb = path.stat().st_size / 1e6
        if size_mb > _LARGE_PDF_MB:
            logger.warning(
                "Large PDF (%.0f MB): %s — extraction may be slow", size_mb, path.name
            )
        raw = _extract_pdf_text(path)
        text = "\n".join(ln.strip() for ln in raw.splitlines() if ln.strip())
        if len(text) < _MIN_TEXT_CHARS:
            logger.warning(
                "Skipping %s — only %d chars extracted (scanned / no text layer?)",
                path.name, len(text),
            )
            return []
        return [_make_document(text, extra_info or {})]


def get_meta(file_path):
    """Pull ticker and fiscal year from a filename like 'DAL_10-K_2024.htm', and
    attach the uniform cross-corpus fields so SEC nodes carry the same schema as
    NASA/RBI. company/year stay (back-compat); corpus/title/citation/source_url/doc_id
    are added. Note: airline_10k must be rebuilt to pick these up (it's stale anyway)."""
    name = Path(file_path).stem          # "DAL_10-K_2024"
    parts = name.split("_")
    company, year = parts[0], int(parts[-1])
    title = f"{company} 10-K {year}"
    return {
        "corpus": "sec",
        "company": company,
        "year": year,
        "doc_id": name,
        "title": title,
        "source_url": "",
        "citation": title,
    }


def _load_sidecar(corpus):
    """Read data/raw/<corpus>/_<corpus>_metadata.jsonl into {basename: record}.

    Keyed on Path(rec["file"]).name because the sidecar's "file" paths use Windows
    backslashes and are project-root-relative, while the reader hands us absolute
    paths — only the basename is reliably comparable."""
    data_dir = Path(CORPORA[corpus]["data_dir"])
    sidecar = data_dir / f"_{corpus}_metadata.jsonl"
    by_name = {}
    if not sidecar.exists():
        logger.warning(
            "No sidecar at %s; %s metadata will be derived from filenames", sidecar, corpus
        )
        return by_name
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_name[Path(rec["file"]).name] = rec
    return by_name


def make_corpus_metadata(corpus):
    """Build a file_metadata callable(file_path) for `corpus` that emits the uniform
    schema (corpus/title/source_url/doc_id/citation) plus corpus-specific extras.

    SEC reuses get_meta (filename-derived, no sidecar). NASA/RBI look their record up
    in the sidecar by basename; a file missing from its sidecar falls back to deriving
    title/doc_id from the filename (and is logged)."""
    if corpus == "sec":
        return get_meta
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {list(CORPORA)}")

    sidecar = _load_sidecar(corpus)

    def _meta(file_path):
        name = Path(file_path).name
        stem = Path(file_path).stem
        rec = sidecar.get(name)
        if rec is None:
            logger.warning(
                "%s not in %s sidecar — deriving title/doc_id from filename", name, corpus
            )
            rec = {}

        if corpus == "nasa":
            doc_id = str(rec.get("id", stem))     # accession id; e.g. "20230012889"
            title = rec.get("title") or stem
            return {
                "corpus": "nasa",
                "doc_id": doc_id,
                "title": title,
                "source_url": rec.get("source_url", ""),
                "citation": f"NASA TR {doc_id} — {title}",
                "date": rec.get("date", ""),
            }

        # corpus == "rbi"
        title = rec.get("title") or stem
        return {
            "corpus": "rbi",
            "doc_id": stem,                       # rbi_N
            "title": title,
            "source_url": rec.get("source_url", ""),
            "citation": f"RBI — {title}",
            "matched_keyword": rec.get("matched_keyword", ""),
        }

    return _meta


def build_reader(input_dir=DEFAULT_DATA_DIR, limit=None):
    """Construct the directory reader. Pass limit=2 to validate on a subset first."""
    return SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        exclude=["manifest.csv"],            # don't ingest the downloader's manifest
        file_extractor={".htm": HTMLTextReader()},
        file_metadata=get_meta,
        num_files_limit=limit,               # None = read every file
    )


def load_and_chunk(input_dir=DEFAULT_DATA_DIR, limit=None):
    """Load filings and split them into nodes ready for embedding.
    Returns (docs, nodes).

    Unchanged SEC default — retrieve.py's BM25 leg calls this with no args, so its
    signature and behavior are frozen. New corpora go through load_and_chunk_corpus.
    """
    docs = build_reader(input_dir, limit).load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    return docs, nodes


def load_and_chunk_corpus(corpus, limit=None):
    """Load + chunk one corpus (sec | nasa | rbi) into nodes ready for embedding.
    Returns (docs, nodes).

    Resolves the data dir from CORPORA[corpus]. SEC keeps its HTML/XBRL reader; NASA
    and RBI use the PDF reader with required_exts=['.pdf'] so the _<corpus>_metadata.jsonl
    sidecar is never ingested as a document. All corpora share the same
    SentenceSplitter(CHUNK_SIZE, CHUNK_OVERLAP) so chunking is consistent across them.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {list(CORPORA)}")

    data_dir = CORPORA[corpus]["data_dir"]
    if corpus == "sec":
        reader = build_reader(data_dir, limit)
    else:
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            recursive=True,
            required_exts=[".pdf"],          # never ingest the _<corpus>_metadata.jsonl sidecar
            file_extractor={".pdf": PDFTextReader()},
            file_metadata=make_corpus_metadata(corpus),
            num_files_limit=limit,
        )

    docs = reader.load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    return docs, nodes


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    corpus = sys.argv[1] if len(sys.argv) > 1 else "sec"
    print(f"PDF backend: {PDF_BACKEND}")
    docs, nodes = load_and_chunk_corpus(corpus)

    print(f"[{corpus}] Loaded {len(docs)} documents -> {len(nodes)} nodes")
    print("sample metadata:", nodes[-1].metadata)
    print("sample text:", nodes[-1].text[:300])