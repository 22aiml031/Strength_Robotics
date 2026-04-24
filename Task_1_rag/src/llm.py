# from __future__ import annotations

# import re

# import requests


# SYSTEM_PROMPT = """You are a careful research paper Q&A assistant.
# Answer only from the selected paper context. If the selected paper context does not directly answer the question, say: "The selected paper does not provide enough evidence to answer this question."
# Write exactly one natural paragraph. Do not use bullets, numbered points, headings, or a source-by-source list.
# Do not mention source numbers. Cite evidence inline using section and page only, for example [Methods, p. 4].
# Use clean, connected English words and do not copy broken OCR spacing from the context.
# """


# def clean_text(text: str) -> str:
#     text = text.replace("\u00ad", "").replace("\u200b", "")
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def clean_answer(text: str) -> str:
#     text = clean_text(text)
#     text = re.sub(r"^[-*\d.\s]+", "", text)
#     return text


# def build_context(chunks: list[dict]) -> str:
#     context_blocks = []
#     for chunk in chunks:
#         section = chunk.get("section") or "Unknown section"
#         page_start = chunk.get("page_start")
#         page_end = chunk.get("page_end")
#         page_label = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
#         content = clean_text(chunk["content"])
#         context_blocks.append(
#             f"[Section: {section}; Pages: {page_label}]\n{content}"
#         )
#     return "\n\n".join(context_blocks)


# def answer_with_ollama(
#     base_url: str,
#     model: str,
#     question: str,
#     retrieved_chunks: list[dict],
# ) -> str:
#     context = build_context(retrieved_chunks)
#     user_prompt = f"""Selected paper context:

# {context}

# Question: {question}

# Answer in exactly one simple paragraph. If the context is not clearly about the question, say the selected paper does not provide enough evidence. Use section/page citations only; do not write Source 1, Source 2, bullets, or a source-by-source answer."""

#     response = requests.post(
#         f"{base_url.rstrip('/')}/api/chat",
#         json={
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_prompt},
#             ],
#             "stream": False,
#             "options": {
#                 "temperature": 0.4,
#                 "top_p": 0.9,
#             },
#         },
#         timeout=180,
#     )
#     response.raise_for_status()
#     payload = response.json()
#     return clean_answer(payload.get("message", {}).get("content", ""))

from __future__ import annotations

import re
import requests


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_answer(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^[-*\d.\s]+", "", text)
    return text


def build_context(chunks: list[dict]) -> str:
    """
    Build labeled context string from retrieved Supabase chunks.
    Page 1 chunks are labeled as ABSTRACT for LLM prioritization.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.get("section") or "Unknown section"
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")

        if page_start == page_end:
            page_label = f"p. {page_start}"
        else:
            page_label = f"pp. {page_start}-{page_end}"

        content = clean_text(chunk.get("content", ""))

        # Label Page 1 chunks as ABSTRACT — highest priority for LLM
        if page_start == 1:
            priority = "*** ABSTRACT (Page 1) - READ THIS FIRST ***"
        elif section and any(kw in section.lower() for kw in [
            "conclusion", "result", "finding", "discussion"
        ]):
            priority = f"*** KEY SECTION: {section} ***"
        else:
            priority = f"Chunk {i}"

        context_blocks.append(
            f"[{priority} | Section: {section} | Pages: {page_label}]\n{content}"
        )

    return "\n\n---\n\n".join(context_blocks)


def _detect_question_type(question: str) -> str:
    """Detect question type for specialized prompt instructions."""
    q = question.lower()

    if any(kw in q for kw in [
        "injury", "risk", "kinematic", "variable", "matter",
        "important", "affect", "impact", "ydrop", "ymax", "x1", "xnet"
    ]):
        return "kinematic"

    if any(kw in q for kw in [
        "tracker", "tracking", "best tracker", "performed best",
        "mediaflow", "kcf", "dlib", "failure rate", "jitter",
        "compare tracker", "which tracker"
    ]):
        return "tracker"

    if any(kw in q for kw in [
        "how does", "how do", "method", "imputation", "handle",
        "pipeline", "process", "approach", "technique", "missing data",
        "step", "algorithm", "work"
    ]):
        return "method"

    if any(kw in q for kw in [
        "result", "performance", "accuracy", "achieve",
        "outperform", "score", "metric", "comparison",
        "better", "improvement", "mse", "rmse", "r2"
    ]):
        return "result"

    if any(kw in q for kw in [
        "problem", "solve", "what does this paper", "purpose",
        "goal", "aim", "objective", "about", "summary",
        "overview", "what is rsimod", "what is the"
    ]):
        return "problem"

    return "general"


def _get_specialized_instructions(question_type: str, question: str) -> str:
    """Return specialized formatting instructions based on question type."""

    if question_type == "kinematic":
        return """
FOR THIS KINEMATIC/INJURY QUESTION — follow this exact format:

Start with: "The following kinematic variables matter most for injury risk:"

Then list each variable found in the context like this:
1. **Ydrop (Bar Drop):** [what it measures]. Injury link: [how less/more Ydrop affects injury risk]
2. **Ymax (Peak Height):** [what it measures]. Injury link: [how it relates to joint stress]
3. **X1 (Horizontal Setup):** [what it measures]. Injury link: [how excessive deviation causes joint loading]
4. **Xnet (Net Displacement):** [what it measures]. Injury link: [trajectory efficiency and injury]
5. **Performance Score (0-4):** [how it integrates variables to assess injury risk]

For each variable explain BOTH what it measures AND why it matters for injury.
End with: "Source: Page X"
Only include variables that are mentioned in the context above.
"""

    if question_type == "tracker":
        return """
FOR THIS TRACKER COMPARISON QUESTION — follow this exact format:

**Best Tracker:** [tracker name] performed best because [specific technical reason].

**Performance Comparison:**
- MedianFlow: failure rate = [X]%, jitter = [X] cm, score agreement = [X]%
- KCF: failure rate = [X]%, jitter = [X] cm, score agreement = [X]%
- Dlib: failure rate = [X]%, jitter = [X] cm, score agreement = [X]%

**Why [best tracker] won:**
[2-3 specific technical reasons from the context]

**Effect of YOLO initialization:**
[How YOLO improved or changed tracker performance]

Source: [Table number, Page number]

IMPORTANT: Use ONLY the exact numbers from the context above.
If a number is not in the context, write "not specified" instead of guessing.
"""

    if question_type == "method":
        return """
FOR THIS METHOD/HOW QUESTION — explain as a numbered pipeline:

Start with: "The imputation method handles missing athlete data through the following steps:"

Step 1: **[Step Name]** — [what happens, which algorithm is used]
Step 2: **[Step Name]** — [what happens, which algorithm is used]
Step 3: **[Step Name]** — [what happens, which algorithm is used]
Step 4: **[Step Name]** — [what happens, which algorithm is used]
[add more steps if found in context]

**Final Result:** [What the complete pipeline achieves with specific MSE/RMSE/R2 numbers]

Source: Page [X]
"""

    if question_type == "result":
        return """
FOR THIS RESULTS/PERFORMANCE QUESTION — present as a comparison:

Start with: "The proposed method achieved the following results:"

**Proposed Method:** MSE=[X], RMSE=[X], R2=[X]
**vs KNN:** RMSE=[X], MAE=[X]
**vs MICE:** RMSE=[X], MAE=[X]
**vs XGBoost:** RMSE=[X], MAE=[X]
**vs EM:** RMSE=[X], MAE=[X]
**vs CART:** RMSE=[X], MAE=[X]

**Key Improvement:** [X]% better than best baseline in [metric]
**External Validation:** [RMSE on external dataset with participant count]

Source: Table [X], Page [X]

Only include rows where numbers appear in the context above.
"""

    if question_type == "problem":
        return """
FOR THIS PROBLEM/OVERVIEW QUESTION — follow this structure:

**Direct Answer:** Start with "This paper solves..." or "This paper addresses..."
State the specific problem in 1-2 sentences.
NEVER start with "Cohort studies..." or "In recent years..."

**What the paper proposes:** [the specific method or system proposed]

**Key Evidence:** Give 3-4 specific facts with numbers:
- Dataset: [number of athletes, weeks, features]
- Problem scale: [missing data rate percentage]  
- Performance: [MSE reduction %, R2 increase %, or RMSE value]
- Comparison: [outperforms which methods by how much]

**Source:** Page [X] (Abstract or Introduction)
"""

    # General fallback
    return """
Answer this question directly and specifically:

**Direct Answer:** [1-2 sentences answering the question directly]

**Supporting Evidence:** [2-3 specific facts with numbers from the context]

**Source:** Page [X]
"""


def answer_with_ollama(
    base_url: str,
    model: str,
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate a detailed grounded answer using Ollama gemma3:4b.
    Uses question-type detection to give specialized formatting instructions.

    Args:
        base_url: Ollama host URL e.g. http://localhost:11434
        model: Model name e.g. gemma3:4b
        question: User question string
        retrieved_chunks: List of chunk dicts from Supabase

    Returns:
        Formatted answer string from Ollama
    """
    context = build_context(retrieved_chunks)
    question_type = _detect_question_type(question)
    specialized = _get_specialized_instructions(question_type, question)

    prompt = f"""You are a research paper analyst.
Answer using ONLY the paper excerpts provided below.
Never use outside knowledge. Never guess or hallucinate numbers.

CRITICAL READING ORDER:
1. Read chunks labeled "*** ABSTRACT (Page 1) ***" FIRST
2. Then read chunks labeled "*** KEY SECTION ***"
3. Then read remaining chunks

PAPER EXCERPTS:
{context}

{'='*60}
QUESTION: {question}
{'='*60}

{specialized}

STRICT RULES — follow all of these:
- Use ONLY information explicitly stated in the excerpts above
- If a specific number is not in the excerpts, write "not specified" — never invent numbers
- Always cite page numbers at the end: "Source: Page X" or "Source: Table X, Page X"
- Never use the word "missingness" — use "missing data rate" or "proportion of missing data" instead
- Never start your answer with "Cohort studies..." or "In recent years..."
- Keep answer between 150 and 300 words
- Use bold text for variable names and key terms
- If information is genuinely not in the excerpts say: "The selected paper does not provide enough evidence to answer this question."

YOUR DETAILED ANSWER:"""

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise research paper analyst. "
                        "Answer questions using only the provided paper excerpts. "
                        "Never hallucinate facts or numbers. "
                        "Always use the exact formatting structure requested. "
                        "Always cite page numbers."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 1024,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    raw_answer = payload.get("message", {}).get("content", "")
    return clean_answer(raw_answer)