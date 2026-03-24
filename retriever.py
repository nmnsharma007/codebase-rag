"""
Advanced retrieval with multi-query + ensemble + re-ranking
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

GEMINI_API_KEY = "key"
genai.configure(api_key=GEMINI_API_KEY)

def analyze_query(query):
    """
    Detect if query asks about specific file.
    Returns: (filename_filter, is_file_query)
    """
    query_lower = query.lower()
    
    # File keyword mapping
    file_patterns = {
        'readme': 'README.md',
        'history': 'HISTORY.md',
        'changelog': 'HISTORY.md',
        'setup': 'setup.py',
        'init': '__init__.py',
        '__init__': '__init__.py',
        'conf.py': 'conf.py',
        'config': 'conf.py'
    }
    
    # Check for file mentions
    for keyword, filename in file_patterns.items():
        if keyword in query_lower:
            print(f"[Query Analyzer] Detected file-specific query: {filename}")
            return filename, True
    
    # No file detected
    print("[Query Analyzer] General semantic query")
    return None, False

class AdvancedRetriever:
    """
    Multi-stage retrieval:
    1. Multi-query: Generate alternative queries
    2. Ensemble: Combine vector + BM25 keyword search
    3. Re-rank: Score results by relevance
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=GEMINI_API_KEY
        )
        
        # Get all documents for BM25
        print("Initializing BM25 retriever...")
        all_data = vector_store.get()
        
        # Reconstruct documents
        from langchain.schema import Document
        self.all_docs = [
            Document(
                page_content=content,
                metadata=metadata
            )
            for content, metadata in zip(all_data['documents'], all_data['metadatas'])
        ]
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        self.bm25_retriever.k = 10  # Get more for re-ranking
        
        # Create vector retriever
        self.vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )
        
        # Ensemble: 60% semantic, 40% keyword
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]
        )
        
        print("✓ Advanced retriever initialized")
    
    def generate_alternative_queries(self, original_query):
        """
    Generate query variations using simple heuristics
    (No LLM calls to avoid rate limits)
    """
        alternatives = []
        
        # Heuristic 1: Add "explain" if not present
        if "explain" not in original_query.lower():
            alternatives.append(f"Explain {original_query}")
        
        # Heuristic 2: Rephrase file queries
        if "readme" in original_query.lower():
            alternatives.append("documentation overview main file")
            alternatives.append("project description and usage")
        elif "how does" in original_query.lower():
            # "How does X work" → "X implementation"
            topic = original_query.lower().replace("how does", "").replace("work", "").strip()
            alternatives.append(f"{topic} implementation")
        
        # Heuristic 3: Add "code" keyword for technical queries
        if any(word in original_query.lower() for word in ["function", "class", "method"]):
            alternatives.append(f"{original_query} code example")
        
        all_queries = [original_query] + alternatives[:2]  # Max 3 total
        
        print(f"[Multi-query] Query variations: {len(all_queries)}")
        for i, q in enumerate(all_queries, 1):
            print(f"  {i}. {q}")
        
        return all_queries
    
    def retrieve_with_ensemble(self, queries):
        """
        Ensemble retrieval: Get docs for all query variations
        """
        all_docs = []
        seen_content = set()
        
        for query in queries:
            docs = self.ensemble_retriever.get_relevant_documents(query)
            
            for doc in docs:
                # Deduplicate
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        print(f"[Ensemble] Retrieved {len(all_docs)} unique documents")
        return all_docs
    
    def rerank_documents(self, query, docs, top_k=5):
        """
        Re-ranking using cross-encoder model (local, no API calls)
        Cross-encoders score query-document pairs directly
        """ 
        if len(docs) <= top_k:
            return docs
        
        print(f"[Re-ranking] Scoring {len(docs)} documents with cross-encoder...")
        
        # Lazy load cross-encoder (only loads once)
        if not hasattr(self, 'cross_encoder'):
            from sentence_transformers import CrossEncoder
            print("[Re-ranking] Loading cross-encoder model (one-time, ~5 seconds)...")
            self.cross_encoder = CrossEncoder('mixedbread-ai/mxbai-rerank-base-v1')
            print("[Re-ranking] Model loaded")
        
        # Prepare pairs: (query, doc_text)
        pairs = [[query, doc.page_content[:512]] for doc in docs]  # Limit to 512 chars
        
        # Score all pairs
        scores = self.cross_encoder.predict(pairs)
        
        # Combine docs with scores
        doc_scores = list(zip(scores, docs))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Get top scores for display
        top_scores = [float(s) for s, _ in doc_scores[:5]]
        print(f"[Re-ranking] Top scores: {[f'{s:.3f}' for s in top_scores]}")
        
        # Return top k documents
        top_docs = [doc for score, doc in doc_scores[:top_k]]
        
        print(f"[Re-ranking] Selected top {top_k} documents")
        
        return top_docs
    
    def retrieve(self, query, top_k=5):
        """
        Full pipeline with query understanding:
        - File-specific queries → metadata filtering
        - General queries → multi-query + ensemble + re-rank
        """
        print(f"\n[Retrieval Pipeline] Query: {query}")
        
        # Analyze query
        filename_filter, is_file_query = analyze_query(query)
        
        if is_file_query:
            # File-specific query - use metadata filtering
            print(f"[Pipeline] Using metadata filter for: {filename_filter}")
            
            # Get all documents
            all_data = self.vector_store.get()
            
            # Filter by filename
            from langchain.schema import Document
            filtered_docs = []
            
            for i, metadata in enumerate(all_data['metadatas']):
                source = metadata.get('source', '')
                if filename_filter.lower() in source.lower():
                    doc = Document(
                        page_content=all_data['documents'][i],
                        metadata=metadata
                    )
                    filtered_docs.append(doc)
            
            print(f"[Pipeline] Found {len(filtered_docs)} chunks from {filename_filter}")
            
            # Return first top_k chunks from that file
            final_docs = filtered_docs[:top_k]
            
        else:
            # General query - full pipeline
            # Stage 1: Multi-query
            queries = self.generate_alternative_queries(query)
            
            # Stage 2: Ensemble retrieval
            docs = self.retrieve_with_ensemble(queries)
            
            # Stage 3: Re-ranking
            final_docs = self.rerank_documents(query, docs, top_k=top_k)
        
        print(f"[Pipeline] Final: {top_k} documents\n")
        
        return final_docs


def load_vector_store():
    """Load vector store"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vector_store