from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


SECTION_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|introduction|background|methods?|methodology|"
    r"experiments?|results?|discussion|conclusion|references|appendix|limitations?|"
    r"related work|data|evaluation|analysis)\b.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    filename: str
    path: Path
    total_pages: int


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    paper_title: str
    chunk_index: int
    content: str
    section: str
    page_start: int
    page_end: int
    source_type: str = "paper_text"
    link_url: str | None = None

    @property
    def metadata(self) -> dict:
        metadata = {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_type": self.source_type,
        }
        if self.link_url:
            metadata["link_url"] = self.link_url
        return metadata


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_url(url: str) -> str:
    url = normalize_text(url)
    url = url.replace(" ", "")
    url = url.rstrip(".,;)]")
    return url


def detect_title(doc: fitz.Document, fallback: str) -> str:
    metadata_title = (doc.metadata or {}).get("title") or ""
    if metadata_title.strip() and "SN Computer Science" not in metadata_title:
        return metadata_title.strip()

    first_page_text = normalize_text(doc[0].get_text("text", sort=True)) if doc.page_count else ""
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    title_lines: list[str] = []
    collecting = False

    for line in lines:
        lower = line.lower()
        if lower == "original research":
            collecting = True
            continue
        if not collecting:
            continue
        if lower.startswith(("received:", "abstract", "keywords")) or "@" in line or "·" in line:
            break
        if lower.startswith(("http", "sn computer science")):
            continue
        title_lines.append(line)

    if title_lines:
        return " ".join(title_lines).strip()

    for line in lines:
        candidate = line.strip()
        if (
            20 <= len(candidate) <= 220
            and not candidate.lower().startswith(("abstract", "keywords", "sn computer science"))
            and "@" not in candidate
        ):
            return candidate
    return fallback


def current_section(text: str, previous: str) -> str:
    for line in text.splitlines():
        clean = line.strip(" .:-")
        if len(clean) > 90:
            continue
        match = SECTION_RE.match(clean)
        if match:
            return match.group(1).strip()
    return previous


def extract_page_links(page: fitz.Page) -> list[str]:
    links: list[str] = []
    for link in page.get_links():
        uri = link.get("uri")
        if uri:
            links.append(normalize_url(uri))

    page_text = normalize_text(page.get_text("text", sort=True))
    links.extend(normalize_url(match.group(0)) for match in re.finditer(r"https?://\S+", page_text))

    seen = set()
    unique_links = []
    for link in links:
        if not link or link in seen:
            continue
        seen.add(link)
        unique_links.append(link)
    return unique_links


def read_paper(path: Path, paper_id: str) -> tuple[Paper, list[dict]]:
    doc = fitz.open(path)
    title = detect_title(doc, paper_id.replace("_", " ").title())
    paper = Paper(
        paper_id=paper_id,
        title=title,
        filename=path.name,
        path=path,
        total_pages=doc.page_count,
    )

    pages = []
    section = "Unknown section"
    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = normalize_text(page.get_text("text", sort=True))
        links = extract_page_links(page)
        if not text and not links:
            continue
        if text:
            section = current_section(text, section)
        pages.append(
            {
                "page": page_number + 1,
                "text": text,
                "section": section,
                "links": links,
            }
        )
    doc.close()
    return paper, pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than chunk_overlap")

    def hard_split(piece: str) -> list[str]:
        if len(piece) <= chunk_size:
            return [piece]
        parts = []
        step = chunk_size - overlap
        for start in range(0, len(piece), step):
            part = piece[start : start + chunk_size].strip()
            if part:
                parts.append(part)
        return parts

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]

        for sentence in sentences:
            pieces = hard_split(sentence.strip())
            for piece in pieces:
                if not piece:
                    continue
                if current and len(current) + len(piece) + 1 > chunk_size:
                    chunks.append(current.strip())
                    current = current[-overlap:].strip()
                    if len(current) + len(piece) + 1 > chunk_size:
                        current = ""
                current = f"{current} {piece}".strip()

    if current:
        chunks.append(current.strip())
    return chunks


def chunk_paper(path: Path, paper_id: str, chunk_size: int, overlap: int) -> tuple[Paper, list[Chunk]]:
    paper, pages = read_paper(path, paper_id)
    chunks: list[Chunk] = []

    seen_links = set()
    for page in pages:
        if page["text"]:
            page_chunks = split_text(page["text"], chunk_size, overlap)
            for content in page_chunks:
                chunks.append(
                    Chunk(
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        chunk_index=len(chunks),
                        content=content,
                        section=page["section"],
                        page_start=page["page"],
                        page_end=page["page"],
                    )
                )

        for link in page.get("links", []):
            link_key = (page["page"], link)
            if link_key in seen_links:
                continue
            seen_links.add(link_key)
            chunks.append(
                Chunk(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    chunk_index=len(chunks),
                    content=(
                        f"Reference link found in {paper.title} on page {page['page']}. "
                        f"URL: {link}. Use this as an external source link cited by the paper."
                    ),
                    section="Reference links",
                    page_start=page["page"],
                    page_end=page["page"],
                    source_type="reference_link",
                    link_url=link,
                )
            )

    return paper, chunks
