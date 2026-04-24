import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

SYSTEM_PROMPT = """You are an Exercise Science Coaching Assistant that provides clear, research-grounded advice using multiple retrieved research papers.

STRICT RULES:

1. MULTI-PAPER SYNTHESIS:
- Use insights from ALL provided research papers. Combine findings logically instead of listing them separately.
- Highlight agreements between studies clearly.

2. CLEAR & SIMPLE LANGUAGE:
- Write in simple English so even a beginner can understand.
- Avoid or simply explain complex jargon (e.g., say "load on knees vs hips" instead of "joint moment distribution").

3. STRUCTURED OUTPUT:
Write exactly 4 paragraphs:
- Paragraph 1: Explain the user's problem or question in simple terms.
- Paragraph 2: Explain what research says (combine multiple papers, mention variations like high-bar vs low-bar, trunk angle, bar path).
- Paragraph 3: Explain WHY it happens (biomechanics in simple language, e.g., how trunk lean shifts load).
- Paragraph 4: Give practical, actionable coaching advice (focus on cues like "chest up", "bar over midfoot", "control descent", "engage core").

4. NO HALLUCINATIONS:
- DO NOT invent authors, studies, or facts. ONLY use information from retrieved papers.
- Use balanced language like "may help" or "is generally associated with" instead of absolute claims.
- If something is unclear, say: "Based on available research, this suggests..."

5. CITATIONS:
- Add simple citations like: [Author/Paper Name, Year]. Keep them clean and minimal.

6. COACHING FOCUS:
- Connect trunk angle, bar path, and squat variation to performance and injury risk.
- Your goal is to HELP the athlete improve performance and reduce injury risk.

Tone: Friendly, clear, coaching-oriented, and not overly academic.
"""

def clean_answer(text):
    """
    Cleans unwanted prefixes and ensures proper paragraph spacing.
    """
    text = re.sub(r"(?i)^(Direct Answer|Key Evidence|Answer|Coaching Advice|Recommendation|Synthesis|Step \d+):\s*", "", text)
    text = re.sub(r"(?m)^[-•*\d]+\s*", "", text)
    return text.strip()

def answer_with_ollama(question, context_chunks, multi_paper=False):
    """
    Interfaces with Ollama for simple, research-grounded coaching synthesis.
    """
    if isinstance(context_chunks, str) or not context_chunks:
        return "No relevant research evidence found."
        
    context_text = ""
    for chunk in context_chunks:
        meta = chunk.get('metadata', {})
        paper_name = meta.get('paper_id', 'Research Paper')
        section = meta.get('section', 'unknown')
        page = meta.get('page', '?')
        context_text += f"Source: {paper_name} | Section: {section} | Page: {page}\nText: {chunk.get('text', '')}\n\n"

    user_prompt = f"""
Selected research context:
{context_text}

User Scenario/Query:
{question}

Instruction:
Generate a clean 4-paragraph coaching response using the context above. 
1. Explain the problem simply. 
2. Synthesize research findings. 
3. Explain the biomechanics simply. 
4. Provide actionable cues.

Use simple English and citations like [Author, Year].
"""

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nAnswer:"

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        raw_answer = data.get("response", "").strip()
        
        if "not provide enough evidence" in raw_answer.lower() or "no relevant context" in raw_answer.lower():
             return "No relevant research evidence found."

        return clean_answer(raw_answer)
        
    except requests.exceptions.RequestException as e:
        return f"Error communicating with local Ollama: {str(e)}"

def generate_coaching_advice(query, retrieved_chunks, multi_paper=False):
    """
    Main entry point for generating coaching advice.
    Supports API_MODE=true as a fallback for users without local Ollama setup.
    """
    import os
    if os.getenv("API_MODE") == "true":
        # Fallback mode for quick demos without local LLM
        if not retrieved_chunks or isinstance(retrieved_chunks, str):
            return "No relevant research evidence found for this scenario."
            
        # Extract a few key sentences from chunks to make it look grounded
        context_preview = ""
        for chunk in retrieved_chunks[:2]:
            text = chunk.get('text', '')[:150] + "..."
            context_preview += f"- {text}\n"
            
        return f"""**Problem Analysis:** Based on your description, this scenario appears to involve biomechanical inefficiencies often cited in literature.

**Research Synthesis:** The retrieved research papers discuss these specific issues, focusing on trunk stability and joint loading. {context_preview}

**Biomechanics:** Movement deviations like the one described typically shift the center of mass, increasing strain on secondary muscle groups.

**Coaching Cues:** Focus on maintaining a stable core and even weight distribution. [Note: This is a simulated response in API Mode. For full research-grounded synthesis, please run with local Ollama.]"""

    return answer_with_ollama(query, retrieved_chunks, multi_paper=multi_paper)
