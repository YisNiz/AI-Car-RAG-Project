"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """TODO (step 2): one chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    # 1. Split the document into sections: (heading, lines). The text before the first "## " has heading "".
    sections = [("", [])]
    for line in doc.text.splitlines():
        if line.startswith("# "):          # the title line - skip it, the path line carries the title
            continue
        if line.startswith("## "):         # a new section starts
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)   # add the line to the current section

    # 2. Turn every non-empty section into a chunk, with the heading path line on top.
    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:                       # e.g. no intro text before the first "## "
            continue
        path = f"{doc.title} › {heading}" if heading else doc.title
        meta = dict(doc.meta, section=heading)
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, f"{path}\n\n{body}", meta))
    return chunks
