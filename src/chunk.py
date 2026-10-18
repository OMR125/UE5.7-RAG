import json
import re
from pathlib import Path
from statistics import median

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- paths (this file lives in src/, data/ sits next to src/) ---------------
ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
DOC_JSON = PARSED / "blueprints_p1-580.json"
TABLE_RECORDS = PARSED / "table_records.json"
CHUNKS = PARSED / "chunks.jsonl"

# ---- settings ----------------------------------------------------------------
CHUNK_SIZE = 250                                   # characters
SEPARATORS = ["\n\n", "\n", ".", "?", "!", " ", ""]
MAX_EMBED_CHARS = 2000                             # rough ceiling for a 512-token embedding model

FURNITURE = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
TEXT_LABELS = {
    DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE, DocItemLabel.CODE,
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=0, separators=SEPARATORS,
    keep_separator="end",                          # remove this argument if your version rejects it
    add_start_index=True,                          # moved here from create_documents
)


# ---- helpers -----------------------------------------------------------------
def clean(text) -> str:
    text = str(text).replace("Downloaded from UnrealDocs.net", "")
    return re.sub(r"\s+", " ", text).strip()


def load_document() -> DoclingDocument:
    return DoclingDocument.load_from_json(DOC_JSON)


# ---- text --------------------------------------------------------------------
def text_sections(doc) -> list:
    """Non-table text grouped by heading path: [{"headings", "paragraphs": [(text, page)]}]."""
    sections, heading_stack, paragraphs, last_page = [], [], [], 1

    def flush():
        if paragraphs:
            sections.append({"headings": " > ".join(t for _, t in heading_stack),
                             "paragraphs": list(paragraphs)})
            paragraphs.clear()

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in FURNITURE:
            continue
        prov = getattr(item, "prov", None)
        if prov:
            last_page = prov[0].page_no
        if isinstance(item, TableItem):
            flush()                                # text before and after a table stays apart
        elif label == DocItemLabel.SECTION_HEADER:
            flush()
            while heading_stack and heading_stack[-1][0] >= item.level:
                heading_stack.pop()
            heading_stack.append((item.level, clean(item.text)))
        elif label in TEXT_LABELS:
            text = clean(item.text)
            if text:
                paragraphs.append((text, last_page))
    flush()
    return sections


def page_at(starts: list, offset: int) -> int:
    """Page of the paragraph that contains this character offset."""
    page = starts[0][1]
    for pos, p in starts:
        if pos > offset:
            break
        page = p
    return page


def text_chunks(sections: list) -> list:
    chunks = []
    for s in sections:
        starts, parts, pos = [], [], 0
        for text, page in s["paragraphs"]:
            starts.append((pos, page))
            parts.append(text)
            pos += len(text) + 2                   # the "\n\n" joiner
        body = "\n\n".join(parts)

        for piece in splitter.create_documents([body]):
            start = max(piece.metadata["start_index"], 0)
            end = start + len(piece.page_content)
            pages = sorted({page_at(starts, start), page_at(starts, end - 1)})
            text = f'{s["headings"]}\n{piece.page_content}' if s["headings"] else piece.page_content
            chunks.append({"type": "text", "headings": s["headings"], "pages": pages,
                           "embed_text": text, "return_text": text})
    return chunks


# ---- tables ------------------------------------------------------------------
def table_chunks(records: list) -> list:
    """One chunk per table record: never passed through the text splitter."""
    return [{"type": "table", "headings": r["headings"], "pages": r["pages"],
             "embed_text": r["embed_text"], "return_text": r["return_text"]} for r in records]


# ---- run ---------------------------------------------------------------------
def report(chunks: list) -> None:
    for kind in ("text", "table"):
        sizes = [len(c["embed_text"]) for c in chunks if c["type"] == kind]
        over = sum(s > MAX_EMBED_CHARS for s in sizes)
        print(f"{kind}: {len(sizes)} chunks, median {median(sizes):.0f} chars, "
              f"max {max(sizes)}, {over} over {MAX_EMBED_CHARS}")


def main() -> None:
    doc = load_document()
    records = json.loads(TABLE_RECORDS.read_text(encoding="utf-8"))
    chunks = text_chunks(text_sections(doc)) + table_chunks(records)
    for i, chunk in enumerate(chunks):
        chunk["id"] = f'{chunk["type"]}-{i}'
    with CHUNKS.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    report(chunks)


if __name__ == "__main__":
    main()