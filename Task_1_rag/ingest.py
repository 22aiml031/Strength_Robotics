from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_DIR, get_settings
from src.embeddings import OllamaEmbeddingClient
from src.pdf_processing import chunk_paper
from src.supabase_store import get_client, replace_chunks, upsert_paper


PAPER_FILES = {
    "paper_1": DATA_DIR / "paper_1.pdf",
    "paper_2": DATA_DIR / "paper_2.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk PDFs, embed them, and upload to Supabase.")
    parser.add_argument(
        "--paper",
        choices=[*PAPER_FILES.keys(), "all"],
        default="all",
        help="Paper to ingest.",
    )
    return parser.parse_args()


def ingest_one(paper_id: str, path: Path) -> None:
    settings = get_settings()
    if not path.exists():
        raise FileNotFoundError(f"Missing PDF for {paper_id}: {path}")

    paper, chunks = chunk_paper(
        path=path,
        paper_id=paper_id,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    print(f"{paper_id}: extracted {len(chunks)} chunks from {paper.total_pages} pages")

    embedder = OllamaEmbeddingClient(
        settings.ollama_base_url,
        settings.ollama_embed_model,
        settings.embedding_dim,
    )
    embeddings = embedder.embed_texts([chunk.content for chunk in chunks])
    print(f"{paper_id}: generated {len(embeddings)} embeddings with {settings.ollama_embed_model}")

    client = get_client(settings.supabase_url, settings.supabase_key)
    upsert_paper(client, paper)
    replace_chunks(client, paper_id, chunks, embeddings)
    print(f"{paper_id}: uploaded to Supabase")


def main() -> None:
    args = parse_args()
    selected = PAPER_FILES if args.paper == "all" else {args.paper: PAPER_FILES[args.paper]}
    for paper_id, path in selected.items():
        ingest_one(paper_id, path)


if __name__ == "__main__":
    main()
