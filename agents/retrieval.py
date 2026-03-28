"""
agents/retrieval.py
--------------------
Retrieval Agent — searches the FAISS index for the most relevant
legal document chunks for a given user query.

Uses HuggingFace all-MiniLM-L6-v2 embeddings (same model used in ingest.py).
No API key required. Runs entirely locally.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, TOP_K_RETRIEVAL, TAVILY_API_KEY
from tavily import TavilyClient

class RetrievalAgent:
    def __init__(self):
        print("[RetrievalAgent] Loading FAISS index...")
        embeddings    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)       
        self.vectordb = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[RetrievalAgent] FAISS index loaded successfully.")
        
        # Initialize Web Search
        self.tavily_client = None
        if TAVILY_API_KEY:
            self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            print("[RetrievalAgent] Tavily Web Search integrated.")

    def retrieve(self, query: str) -> list[dict]:
        """
        Search the vector store for the top-K most relevant legal chunks.       
        Also queries the web via Tavily if a key is provided.
        """
        # 1. Local FAISS Search
        results = self.vectordb.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)

        chunks = []
        for doc, score in results:
            chunks.append({
                "content":   doc.page_content,
                "act":       doc.metadata.get("act",      "Unknown Act"),       
                "source":    doc.metadata.get("source",   "Unknown Source"),    
                "law_type":  doc.metadata.get("law_type", "central"),
                "relevance": round(float(score), 4),
            })
            
        # 2. Web Search via Tavily (Dynamic real-time retrieval)
        if self.tavily_client:
            try:
                print(f"[RetrievalAgent] Fetching live web context for: '{query}'...")
                # Ask Tavily to search for context
                web_results = self.tavily_client.search(
                    query=query + " legal implications India law cyber privacy", 
                    search_depth="basic",
                    max_results=2
                )
                
                for i, r in enumerate(web_results.get("results", [])):
                    chunks.append({
                        "content":   r.get("content", ""),
                        "act":       "Live Web Search",
                        "source":    r.get("url", "Web"),
                        "law_type":  "live-article/cyber",
                        "relevance": 0.1  # Highly relevant if it's dynamic
                    })
            except Exception as e:
                print(f"[RetrievalAgent] Web search failed: {e}")
        return chunks