import sys
from agents.retrieval import RetrievalAgent

print("Loading Retrieval Agent and checking FAISS Index...")
r = RetrievalAgent()

print("\n--- TEST: Checking Index Metadata ---")
try:
    docstore = r.vectordb.docstore._dict
    chunks_count = len(docstore)
    acts = set()
    sources = set()
    
    for doc_id, doc in docstore.items():
        if "act" in doc.metadata:
            acts.add(doc.metadata["act"])
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])
            
    print(f"Total FAISS Chunks Found: {chunks_count}")
    print(f"\nUnique Acts/Documents embedded in the index ({len(acts)} found):")
    for act in sorted(acts):
        print(f"  - {act}")
        
    print(f"\nUnique File Sources ({len(sources)} found):")
    for source in sorted(sources):
        print(f"  - {source}")
        
except Exception as e:
    print(f"Could not easily read docstore directly: {e}")

print("\n--- TEST: Search Queries (Local FAISS + Web) ---")

test_queries = [
    "What is the penalty for murder?", # Targets IPC
    "How do I file a consumer complaint for a defective product?", # Targets Consumer Protection
    "What are the grounds for eviction of a tenant by a landlord?", # Targets Rent Control
    "What are the implications of the MOVEit data breach?", # Should target Live Web/Tavily
    "How is a dishonoured cheque dealt with under the law?", # Targets Negotiable Instruments
    "What are the rights of a woman facing domestic abuse?" # Targets Domestic Violence act
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    chunks = r.retrieve(query)
    
    if chunks:
        print(f"  Returned {len(chunks)} relevant chunks.")
        for i, c in enumerate(chunks, 1):
            if c['act'] == "Live Web Search":
                 print(f"    {i}. {c['act']} -> {c['source']}")
            else:
                 print(f"    {i}. {c['act']} -> {c['source']} (Score/Distance: {c['relevance']})")
    else:
        print("  Found 0 chunks.")
        
print("\nTests completed.")