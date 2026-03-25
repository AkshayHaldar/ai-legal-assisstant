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
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, TOP_K_RETRIEVAL


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

    def retrieve(self, query: str) -> list[dict]:
        """
        Search the vector store for the top-K most relevant legal chunks.

        Returns a list of dicts:
        [
          {
            "content":   "...text of the chunk...",
            "act":       "Indian Penal Code",
            "source":    "ipc.pdf",
            "relevance": 0.312   (lower FAISS score = more similar)
          },
          ...
        ]
        """
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

        return chunks
