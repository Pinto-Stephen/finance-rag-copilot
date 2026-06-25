"""
ingest.py — Load airline 10-K filings, strip their HTML/XBRL to clean text,
attach company/year metadata, and split into chunks (nodes) ready for embedding.

Run directly from the project root to sanity-check the load/chunk step:
    python -m src.ingest

Import it from index_build.py to get the nodes without side effects:
    from src.ingest import load_and_chunk
    docs, nodes = load_and_chunk()
"""

import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_DATA_DIR

# 10-Ks are XBRL parsed as HTML, so this warning would otherwise fire constantly.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Edit if you run from a different working directory, or move this to
# config.settings as DATA_DIR and import it.



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
        return [Document(text=text, metadata=extra_info or {})]


def get_meta(file_path):
    """Pull ticker and fiscal year from a filename like 'DAL_10-K_2024.htm'."""
    name = Path(file_path).stem          # "DAL_10-K_2024"
    parts = name.split("_")
    return {"company": parts[0], "year": int(parts[-1])}


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
    """
    docs = build_reader(input_dir, limit).load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    return docs, nodes


if __name__ == "__main__":

    docs, nodes = load_and_chunk()

    print(f"Loaded {len(docs)} documents -> {len(nodes)} nodes")
    print("sample metadata:", nodes[-1].metadata)      # expect {'company': 'DAL', 'year': 2024}
    print("sample text:", nodes[-1].text[:300])