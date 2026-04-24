-- Demo/development policies for using a Supabase publishable/anon key.
-- Run this in the Supabase SQL editor after sql/schema.sql.
--
-- For production, prefer a server-side service role key for ingestion and
-- narrower read-only policies for the Streamlit app.

alter table public.papers enable row level security;
alter table public.paper_chunks enable row level security;

drop policy if exists "demo_public_read_papers" on public.papers;
drop policy if exists "demo_public_write_papers" on public.papers;
drop policy if exists "demo_public_read_chunks" on public.paper_chunks;
drop policy if exists "demo_public_write_chunks" on public.paper_chunks;

create policy "demo_public_read_papers"
on public.papers
for select
to anon, authenticated
using (true);

create policy "demo_public_write_papers"
on public.papers
for all
to anon, authenticated
using (true)
with check (true);

create policy "demo_public_read_chunks"
on public.paper_chunks
for select
to anon, authenticated
using (true);

create policy "demo_public_write_chunks"
on public.paper_chunks
for all
to anon, authenticated
using (true)
with check (true);

grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.papers to anon, authenticated;
grant select, insert, update, delete on public.paper_chunks to anon, authenticated;
grant usage, select on sequence public.paper_chunks_id_seq to anon, authenticated;
grant execute on function public.match_paper_chunks(vector, int, text) to anon, authenticated;
