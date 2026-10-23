"""Hybrid retrieval with LangChain: BM25 + dense search merged with EnsembleRetriever (RRF)."""
import json
import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "parsed" / "chunks.jsonl"
DB_PATH = ROOT / "chroma_db"
COLLECTION = "unreal_blueprints"              
EMBED_MODEL = "BAAI/bge-small-en-v1.5"        
CANDIDATES = 30                               
TOP_K = 5


def load_documents() -> list:
    """chunks.jsonl -> LangChain Documents (return_text is the searchable text)."""
    with CHUNKS.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return [Document(page_content=c["return_text"],
                     metadata={"type": c["type"], "headings": c["headings"],
                               "pages": ",".join(map(str, c["pages"]))})
            for c in chunks]


def build_retrievers() -> dict:
    bm25 = BM25Retriever.from_documents(
        load_documents(), k=CANDIDATES,
        preprocess_func=lambda text: re.findall(r"\w+", text.lower()),   # lowercase, no punctuation
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    store = Chroma(client=chromadb.PersistentClient(path=str(DB_PATH)),
                   collection_name=COLLECTION, embedding_function=embeddings)
    dense = store.as_retriever(search_kwargs={"k": CANDIDATES})

    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.3, 0.7])
    return {"keyword": bm25, "dense": dense, "hybrid": hybrid}


class Retriever:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.retrievers = build_retrievers()

    def search(self, query: str, mode: str = "hybrid", top_k: int = None) -> list:
        """mode: 'dense' | 'keyword' | 'hybrid'."""
        docs = self.retrievers[mode].invoke(query)[: top_k or self.top_k]
        return [{"type": d.metadata["type"], "headings": d.metadata["headings"],
                 "pages": [int(p) for p in d.metadata["pages"].split(",")],
                 "return_text": d.page_content} for d in docs]


MODES = ("dense", "keyword", "hybrid")
SNIPPET_CHARS = 200


def show_hits(question: str, mode: str, hits: list) -> None:
    print(f"\n[{mode}] {question}")
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit["return_text"].split())[:SNIPPET_CHARS]
        print(f'  {rank}. {hit["type"]} | pages {hit["pages"]} | {hit["headings"]}')
        print(f"     {snippet}")


def ask(retriever: Retriever, questions: list, modes: tuple = MODES) -> None:
    """Run every question through every mode and print the hits."""
    for question in questions:
        for mode in modes:
            show_hits(question, mode, retriever.search(question, mode))