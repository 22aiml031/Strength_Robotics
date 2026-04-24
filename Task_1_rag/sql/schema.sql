create extension if not exists vector;

create table if not exists public.papers (
  id text primary key,
  title text not null,
  filename text not null,
  total_pages integer not null,
  created_at timestamptz not null default now()
);

create table if not exists public.paper_chunks (
  id bigserial primary key,
  paper_id text not null references public.papers(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  section text,
  page_start integer,
  page_end integer,
  token_count integer,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(384) not null,
  created_at timestamptz not null default now(),
  unique (paper_id, chunk_index)
);

create index if not exists paper_chunks_paper_id_idx
  on public.paper_chunks (paper_id);

create index if not exists paper_chunks_embedding_idx
  on public.paper_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create or replace function public.match_paper_chunks(
  query_embedding vector(384),
  match_count int,
  filter_paper_id text
)
returns table (
  id bigint,
  paper_id text,
  chunk_index integer,
  content text,
  section text,
  page_start integer,
  page_end integer,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    pc.id,
    pc.paper_id,
    pc.chunk_index,
    pc.content,
    pc.section,
    pc.page_start,
    pc.page_end,
    pc.metadata,
    1 - (pc.embedding <=> query_embedding) as similarity
  from public.paper_chunks pc
  where pc.paper_id = filter_paper_id
  order by pc.embedding <=> query_embedding
  limit match_count;
$$;
