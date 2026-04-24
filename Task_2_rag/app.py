import streamlit as st
import os
from ingest import build_index
from retriever import Retriever
from generator import generate_coaching_advice

# Set up page layout
st.set_page_config(page_title="Exercise Science Coaching Agent", layout="wide")

st.title("Exercise Science Coaching Agent 🏋️‍♂️")
st.write("Get research-grounded coaching cues for exercise performance based on literature.")

# Initialize the Retriever using session_state (re-init if stale)
if "retriever" not in st.session_state or not hasattr(st.session_state.retriever, "retrieve_chunks"):
    st.session_state.retriever = Retriever()

# Sidebar for Admin Controls
with st.sidebar:
    st.header("Admin Controls")
    st.write("Use this button to process new PDFs placed in `data/papers/` and push to Supabase.")
    if st.button("Rebuild Index"):
        with st.spinner("Extracting PDFs and building Supabase index..."):
            try:
                build_index()
                st.success("Index rebuilt successfully in Supabase!")
            except Exception as e:
                st.error(f"Error rebuilding index: {e}")

# Main application logic
DATA_DIR = "/Users/devangpatel/Strength Robotics/Task_2_rag/data"
if os.path.exists(DATA_DIR):
    available_papers = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
else:
    available_papers = []

if not available_papers:
    st.warning("No papers found in `data/papers/`. Please add PDFs and rebuild the index.")
else:
    # 1. Search Mode Selection
    search_mode = st.radio(
        "Select Search Mode:",
        ["Single Paper", "All Papers"]
    )

    # 2. Conditional Paper Selection
    if search_mode == "Single Paper":
        paper_id = st.selectbox("Select a paper to search:", available_papers)
    else:
        st.info("Searching across all research papers")
        paper_id = None
    
    query = st.text_area(
        "Enter athlete scenario or query:",
        placeholder="e.g., Athlete performed 5 reps squat at 80%, form score 6/10, bar drifting forward.",
        height=100
    )
    
    if st.button("Get Coaching Advice", type="primary"):
        if not query.strip():
            st.warning("Please enter an athlete scenario or query first.")
        else:
            with st.spinner("Retrieving research and generating advice..."):
                # Step 1: Retrieve chunks (supporting both search modes)
                if search_mode == "Single Paper":
                    retrieved_chunks = st.session_state.retriever.retrieve_chunks(query, paper_id, top_k=7, threshold=0.4)
                else:
                    retrieved_chunks = st.session_state.retriever.retrieve_chunks(query, None, threshold=0.4)
                
                # Step 2: Generate response via LLM
                advice = generate_coaching_advice(query, retrieved_chunks, multi_paper=(search_mode == "All Papers"))
                
                # Step 3: Output Results
                st.subheader("Final Coaching Advice")
                st.info(advice)
                
                st.divider()
                
                # Output retrieved chunks
                st.subheader("Top Retrieved Chunks")
                if isinstance(retrieved_chunks, str):
                    st.warning(retrieved_chunks)
                else:
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        meta = chunk['metadata']
                        header_text = f"Source {i} | {meta['paper_id']} | {meta['section']} (Page {meta['page']})"
                        with st.expander(header_text):
                            st.write(chunk['text'])
