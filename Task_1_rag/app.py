from __future__ import annotations

import re
import streamlit as st

from src.config import get_settings
from src.embeddings import OllamaEmbeddingClient
from src.llm import answer_with_ollama
from src.supabase_store import (
    get_client,
    list_papers,
    list_reference_links,
    match_chunks,
)

st.set_page_config(
    page_title="Research Paper Q&A Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def clients():
    settings = get_settings()
    supabase = get_client(settings.supabase_url, settings.supabase_key)
    embedder = OllamaEmbeddingClient(
        settings.ollama_base_url,
        settings.ollama_embed_model,
        settings.embedding_dim,
    )
    return settings, supabase, embedder


def is_reference_question(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in [
        "reference", "references", "link", "links",
        "url", "urls", "doi", "source link"
    ])


def format_reference_links(rows: list[dict]) -> str:
    if not rows:
        return "No reference links found for this paper."
    links = []
    seen = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        url = metadata.get("link_url")
        if not url or url in seen:
            continue
        if "orcid" in url:
            continue
        seen.add(url)
        page = row.get("page_start", "?")
        links.append(f"- [{url}]({url}) (page {page})")
    if not links:
        return "No reference links found for this paper."
    return "### Source Links\n\n" + "\n".join(links)


def expand_question_for_retrieval(question: str) -> str:
    """Expand query with domain-specific keywords for better retrieval."""
    lower = question.lower()

    if any(kw in lower for kw in [
        "tracker", "tracking", "mediaflow", "kcf", "dlib",
        "best tracker", "failure rate", "jitter"
    ]):
        return (
            question
            + " MedianFlow KCF Dlib tracker performance failure rate "
            + "jitter stable robust accurate barbell trajectory score match "
            + "tracking failures robustness analysis table"
        )

    if any(kw in lower for kw in [
        "kinematic", "injury", "ydrop", "ymax", "x1", "xnet",
        "variable", "matter", "risk"
    ]):
        return (
            question
            + " Ydrop Ymax X1 Xnet kinematic variables injury risk "
            + "bar drop catch phase horizontal displacement joint loading "
            + "performance score biomechanical"
        )

    if any(kw in lower for kw in [
        "imputation", "missing", "handle", "how does", "method",
        "pipeline", "approach"
    ]):
        return (
            question
            + " feature importance factor analysis clustering XGBoost "
            + "softmax similarity score imputed value weighted average "
            + "missing value imputation pipeline steps"
        )

    if any(kw in lower for kw in [
        "problem", "solve", "objective", "goal", "purpose",
        "rsimod", "what is", "about"
    ]):
        return (
            question
            + " missing data challenge cohort studies proposed methodology "
            + "achieved reduction MSE increase R2 RSImod athletic readiness "
            + "novel imputation technique"
        )

    return question


def is_noisy_evidence(chunk: dict) -> bool:
    metadata = chunk.get("metadata") or {}
    if metadata.get("source_type") == "reference_link":
        return False
    content = (chunk.get("content") or "").lower()
    page_start = chunk.get("page_start") or 0
    if page_start >= 15 and (
        "doi" in content or
        "http" in content or
        "reference" in content
    ):
        return True
    return False


def clean_retrieved_chunks(
    chunks: list[dict],
    keep_count: int
) -> list[dict]:
    """
    Clean noisy chunks but always preserve Page 1 chunks (Abstract).
    Page 1 chunks are injected at the front to guarantee Abstract context.
    """
    # Separate Page 1 chunks from others
    page1_chunks = [c for c in chunks if c.get("page_start") == 1]
    other_chunks = [
        c for c in chunks
        if c.get("page_start") != 1 and not is_noisy_evidence(c)
    ]

    # Always include up to 2 Page 1 chunks + fill rest from similarity results
    mandatory = page1_chunks[:2]
    remaining_slots = keep_count - len(mandatory)
    remaining = other_chunks[:remaining_slots]

    combined = mandatory + remaining
    return combined if combined else chunks[:keep_count]


def normalize_answer_text(answer: str) -> str:
    answer = answer.replace("\u00ad", "").replace("\u200b", "")
    answer = re.sub(r"\s+", " ", answer).strip()
    replacements = {
        "sco re": "score",
        "perform ance": "performance",
        "effi cient": "efficient",
        "ath lete": "athlete",
        "hi gh": "high",
        "qual itative": "qualitative",
        "d ynamic": "dynamic",
        "ru le-based": "rule-based",
        "Yd rop": "Ydrop",
        "dri ven": "driven",
        "sta ble": "stable",
        "robust ness": "robustness",
        "initial ization": "initialization",
        "missingness": "missing data rate",
        "Missingness": "Missing data rate",
    }
    for bad, good in replacements.items():
        answer = answer.replace(bad, good)
    return answer


def render_answer(answer: str) -> None:
    answer = normalize_answer_text(answer)
    st.markdown(answer)


# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("📚 Research Paper Q&A Agent")

try:
    settings, supabase, embedder = clients()
    papers = list_papers(supabase)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if not papers:
    st.warning(
        "No papers found in Supabase yet. "
        "Run `python3 ingest.py --paper all` first."
    )
    st.stop()

paper_options = {
    f"{paper['id']} - {paper['title']}": paper["id"]
    for paper in papers
}

with st.sidebar:
    st.header("📄 Select Paper")
    selected_label = st.selectbox(
        "Paper",
        options=list(paper_options.keys())
    )
    selected_paper_id = paper_options[selected_label]

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown(
        f"- **LLM:** Ollama (`{settings.ollama_chat_model}`) — Local\n"
        f"- **Embeddings:** `{settings.ollama_embed_model}`\n"
        "- **Vector DB:** Supabase (pgvector)\n"
        "- **UI:** Streamlit"
    )

    st.markdown("---")
    st.subheader("Example Questions")
    st.caption("Paper 1 (Imputation):")
    st.code("What problem does this paper solve?", language=None)
    st.code("How does imputation handle missing athlete data?", language=None)
    st.code("What is RSImod and how is it used?", language=None)

    st.caption("Paper 2 (Weightlifting):")
    st.code("What tracker performed best and why?", language=None)
    st.code("What kinematic variables matter most for injury risk?", language=None)

# Main panel
st.markdown(f"**Currently querying:** `{selected_label}`")
st.markdown("---")

question = st.text_area(
    "Ask a question about the selected paper:",
    placeholder=(
        "e.g. What problem does this paper solve?\n"
        "e.g. How does the imputation method handle missing athlete data?\n"
        "e.g. What tracker performed best and why?"
    ),
    height=120,
    key="question_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True
    )
with col2:
    clear = st.button("🗑️ Clear", use_container_width=False)

if clear:
    st.session_state.pop("last_answer", None)
    st.session_state.pop("last_chunks", None)
    st.rerun()

if ask and question.strip():
    if is_reference_question(question.strip()):
        rows = list_reference_links(supabase, selected_paper_id)
        answer = format_reference_links(rows)
        st.subheader("Answer")
        render_answer(answer)

    else:
        with st.spinner(
            "🔎 Retrieving relevant sections... "
            "🤖 Generating answer with Ollama (gemma3:4b)..."
        ):
            try:
                retrieval_question = expand_question_for_retrieval(
                    question.strip()
                )
                query_embedding = embedder.embed_query(retrieval_question)

                raw_chunks = match_chunks(
                    supabase,
                    selected_paper_id,
                    query_embedding,
                    max(settings.match_count * 3, 20),
                )

                chunks = clean_retrieved_chunks(
                    raw_chunks,
                    settings.match_count
                )

                if not chunks:
                    st.info(
                        "No matching evidence found for this paper. "
                        "Try rephrasing your question."
                    )
                    st.stop()

                answer = answer_with_ollama(
                    settings.ollama_base_url,
                    settings.ollama_chat_model,
                    question.strip(),
                    chunks,
                )

                st.session_state["last_answer"] = answer
                st.session_state["last_chunks"] = chunks

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        # Show answer
        st.markdown("---")
        st.subheader("💡 Answer")
        render_answer(answer)

        # Show retrieval quality
        page1_count = sum(
            1 for c in chunks if c.get("page_start") == 1
        )
        total = len(chunks)
        if page1_count > 0:
            st.success(
                f"✅ Retrieval: {page1_count} Abstract chunks included "
                f"| {total} total chunks retrieved"
            )
        else:
            st.warning(
                f"⚠️ No Abstract chunks retrieved. "
                f"{total} chunks used from similarity search."
            )

        # Show source chunks
        with st.expander(
            f"📄 View Source Chunks ({total} retrieved)",
            expanded=False
        ):
            for i, chunk in enumerate(chunks, start=1):
                page = chunk.get("page_start", "?")
                section = chunk.get("section", "Unknown")
                similarity = chunk.get("similarity", 0.0)
                sim_str = (
                    f"{similarity:.3f}"
                    if isinstance(similarity, float)
                    else "N/A"
                )
                is_abstract = page == 1
                badge = "🔒 ABSTRACT" if is_abstract else "🔍 SIMILARITY"
                st.markdown(
                    f"**Chunk {i}** {badge} | "
                    f"Page `{page}` | "
                    f"Section: `{section}` | "
                    f"Similarity: `{sim_str}`"
                )
                preview = (chunk.get("content") or "")[:400]
                if len(chunk.get("content", "")) > 400:
                    preview += "..."
                st.text(preview)
                if i < total:
                    st.markdown("---")