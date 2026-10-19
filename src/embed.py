import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "parsed" / "chunks.jsonl"
DB_PATH = ROOT / "chroma_db"
COLLECTION = "unreal_blueprints"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
ENCODE_BATCH = 64
ADD_BATCH = 1000


def load_chunks() -> list:
    with CHUNKS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL, device="cuda")

    # Chunks longer than the model's limit are silently cut off, so count them first.
    lengths = [len(model.tokenizer(c["embed_text"])["input_ids"]) for c in chunks]
    over = sum(n > model.max_seq_length for n in lengths)
    print(f"max_seq_length {model.max_seq_length}; {over} of {len(chunks)} chunks are longer")

    vectors = model.encode([c["embed_text"] for c in chunks], batch_size=ENCODE_BATCH,
                           normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)          # rebuild from scratch, no duplicates
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), ADD_BATCH):
        part = chunks[i:i + ADD_BATCH]
        collection.add(
            ids=[c["id"] for c in part],
            embeddings=vectors[i:i + ADD_BATCH].tolist(),
            documents=[c["return_text"] for c in part],          # what retrieval returns
            metadatas=[{"type": c["type"], "headings": c["headings"],
                        "pages": ",".join(map(str, c["pages"])),
                        "page_first": c["pages"][0]} for c in part],
        )
    print(collection.count(), "chunks stored in", DB_PATH)


if __name__ == "__main__":
    main()