import os
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if SUPABASE_URL and SUPABASE_KEY:
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            self.supabase = None
            print("Warning: SUPABASE_URL or SUPABASE_KEY not set in environment.")

    def retrieve_chunks(self, query, paper_id=None, top_k=5, threshold=0.4):
        """
        Retrieves relevant chunks. If paper_id is None, searches across all papers.
        """
        if not self.supabase:
            return "No relevant research evidence found."
            
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        try:
            if paper_id:
                # MODE 1: Single Paper Mode (filtered by paper_id)
                response = self.supabase.rpc(
                    "match_documents_by_paper",
                    {
                        "query_embedding": query_embedding.tolist(),
                        "match_threshold": threshold,
                        "match_count": top_k,
                        "paper_filter": paper_id
                    }
                ).execute()
                results = response.data
            else:
                # MODE 2: All Papers Mode (Diversity Strategy)
                # 1. Retrieve initial candidates (k=15)
                response = self.supabase.rpc(
                    "match_all_documents",
                    {
                        "query_embedding": query_embedding.tolist(),
                        "match_threshold": threshold,
                        "match_count": 15
                    }
                ).execute()
                
                all_results = response.data
                if not all_results:
                    return "No relevant research evidence found."
                
                # 2. Group results by paper_id
                paper_groups = {}
                for r in all_results:
                    pid = r.get("paper_id", "Unknown")
                    if pid not in paper_groups:
                        paper_groups[pid] = []
                    paper_groups[pid].append(r)
                
                # Identify unique papers in order of their best similarity score
                pids_ordered = []
                for r in all_results:
                    pid = r.get("paper_id", "Unknown")
                    if pid not in pids_ordered:
                        pids_ordered.append(pid)
                
                final_chunks = []
                # Step A: First pass (ensure diversity) - Take top chunk from each paper (up to 4 papers)
                for pid in pids_ordered[:4]:
                    if paper_groups[pid]:
                        final_chunks.append(paper_groups[pid].pop(0))
                
                # Step B: Second pass (fill remaining slots) - Add additional chunks (max 2 per paper total)
                # until we reach the target of 6-8 chunks
                if len(final_chunks) < 8:
                    for pid in pids_ordered:
                        if paper_groups[pid]:
                            final_chunks.append(paper_groups[pid].pop(0))
                            if len(final_chunks) >= 8:
                                break
                
                # Logging requirement: total unique papers and chunks per paper
                print(f"\n--- Diversity Retrieval Report ---")
                print(f"Total unique papers found in top 15: {len(paper_groups)}")
                paper_counts = {}
                for c in final_chunks:
                    pid = c.get('paper_id', 'Unknown')
                    paper_counts[pid] = paper_counts.get(pid, 0) + 1
                print(f"Chunks per paper in final set: {paper_counts}")
                print(f"---------------------------------\n")
                
                results = final_chunks
            
            if not results:
                return "No relevant research evidence found."
                
            formatted_results = []
            for r in results:
                actual_paper_id = r.get("paper_id") or paper_id
                formatted_results.append({
                    "text": r["content"],
                    "metadata": {
                        "paper_id": actual_paper_id,
                        "section": r["section"],
                        "page": r["page"]
                    }
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Supabase RPC Error: {e}")
            return "No relevant research evidence found."

if __name__ == "__main__":
    retriever = Retriever()
    # Test single paper
    print("Testing Single Paper:")
    print(retriever.retrieve_chunks("Squat cues", "sample.pdf"))
    # Test all papers
    print("\nTesting All Papers:")
    print(retriever.retrieve_chunks("Squat cues"))
