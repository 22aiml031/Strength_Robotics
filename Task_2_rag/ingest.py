import os
import re
import fitz  # PyMuPDF
import numpy as np
import base64
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "/Users/devangpatel/Strength Robotics/Task_2_rag/data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in environment.")

def extract_and_chunk_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paper_id = os.path.basename(pdf_path)
    
    current_section = "unknown"
    
    # Soft section keywords mapping
    section_keywords = {
        "abstract": "abstract",
        "introduction": "introduction",
        "methods": "methods",
        "methodology": "methods",
        "materials and methods": "methods",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion"
    }
    
    # Using specific chunk_size=300 and overlap=80 to drastically increase chunk count
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80
    )
    
    final_chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract full text properly without filtering blocks
        text = page.get_text("text")
        
        # Fallback to OCR if extracted text is too short or empty
        if len(text.strip()) < 50:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            payload = {
                "model": "gemma3:4b",
                "prompt": "Extract all the text from this image exactly as it appears. Do not add any extra commentary, just return the text.",
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            
            try:
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                if response.status_code == 200:
                    text = response.json().get("response", "").strip()
                else:
                    text = ""
            except Exception as e:
                print(f"Ollama OCR Error on page {page_num+1}: {e}")
                text = ""
            
        if not text.strip():
            continue
            
        # Clean text: remove excessive whitespace and broken OCR spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks
        splits = splitter.split_text(text)
        
        for s in splits:
            lower_s = s.lower()
            
            # Simple soft section detection based on keywords in the chunk
            prefix = lower_s[:150]
            for kw, sec_name in section_keywords.items():
                if re.search(r'\b' + kw + r'\b', prefix):
                    current_section = sec_name
                    break
            
            # Append chunk regardless of section detection success
            final_chunks.append({
                "text": s,
                "metadata": {
                    "paper_id": paper_id,
                    "section": current_section,
                    "page": page_num + 1
                }
            })
            
    return final_chunks

def build_index():
    if not supabase:
        print("Cannot build index: Supabase credentials missing.")
        return
        
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist.")
        return
        
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in the data directory.")
        return
        
    total_chunks = 0
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    for pdf_file in pdf_files:
        path = os.path.join(DATA_DIR, pdf_file)
        print(f"\nProcessing {pdf_file}...")
        
        chunks = extract_and_chunk_pdf(path)
        num_chunks = len(chunks)
        
        print(f"Chunks for {pdf_file}: {num_chunks}")
        if num_chunks < 80:
            print(f"WARNING: {pdf_file} produced only {num_chunks} chunks. Expected 80-200.")
            
        if num_chunks == 0:
            continue
            
        # Batch Embeddings
        texts = [c["text"] for c in chunks]
        print(f"Generating embeddings for {pdf_file}...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Supabase Batch Insertion
        batch_size = 100
        inserted_for_paper = 0
        
        for i in range(0, num_chunks, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            insert_data = []
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                meta = chunk["metadata"]
                insert_data.append({
                    "paper_id": meta["paper_id"],
                    "content": chunk["text"],
                    "section": meta["section"],
                    "page": meta["page"],
                    "embedding": emb.tolist()
                })
                
            try:
                supabase.table("documents").insert(insert_data).execute()
                inserted_for_paper += len(batch_chunks)
            except Exception as e:
                print(f"Error inserting batch for {pdf_file}: {e}")
                
        print(f"Inserted {inserted_for_paper} chunks for paper {pdf_file}")
        total_chunks += inserted_for_paper
        
    print(f"\n=========================================")
    print(f"Total chunks inserted across all papers: {total_chunks}")
    print(f"=========================================")

if __name__ == "__main__":
    build_index()
