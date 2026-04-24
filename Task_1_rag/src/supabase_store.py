from __future__ import annotations

from postgrest.exceptions import APIError
from supabase import Client, create_client

from src.pdf_processing import Chunk, Paper


RLS_HELP = (
    "Supabase blocked this write with Row Level Security. Run "
    "sql/demo_rls_policies.sql in the Supabase SQL editor for this demo, "
    "or use SUPABASE_SERVICE_ROLE_KEY for ingestion."
)


def get_client(url: str, key: str) -> Client:
    return create_client(url, key)


def execute_or_explain_rls(request):
    try:
        return request.execute()
    except APIError as exc:
        message = getattr(exc, "message", "") or str(exc)
        if "row-level security" in message.lower() or "42501" in message:
            raise RuntimeError(RLS_HELP) from exc
        raise


def upsert_paper(client: Client, paper: Paper) -> None:
    execute_or_explain_rls(client.table("papers").upsert(
        {
            "id": paper.paper_id,
            "title": paper.title,
            "filename": paper.filename,
            "total_pages": paper.total_pages,
        }
    ))


def replace_chunks(
    client: Client,
    paper_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    batch_size: int = 50,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    execute_or_explain_rls(client.table("paper_chunks").delete().eq("paper_id", paper_id))

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append(
            {
                "paper_id": chunk.paper_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": max(1, len(chunk.content.split())),
                "embedding": embedding,
                "metadata": chunk.metadata,
            }
        )

    for start in range(0, len(rows), batch_size):
        execute_or_explain_rls(
            client.table("paper_chunks").insert(rows[start : start + batch_size])
        )


def list_papers(client: Client) -> list[dict]:
    response = client.table("papers").select("*").order("id").execute()
    return response.data or []


def match_chunks(
    client: Client,
    paper_id: str,
    query_embedding: list[float],
    match_count: int,
) -> list[dict]:
    response = client.rpc(
        "match_paper_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_paper_id": paper_id,
        },
    ).execute()
    return response.data or []


def list_reference_links(client: Client, paper_id: str, limit: int = 20) -> list[dict]:
    response = (
        client.table("paper_chunks")
        .select("content, section, page_start, page_end, metadata")
        .eq("paper_id", paper_id)
        .eq("section", "Reference links")
        .order("page_start")
        .limit(limit)
        .execute()
    )
    return response.data or []
