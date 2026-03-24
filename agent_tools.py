"""
Tools for LangGraph agent
"""

from langchain.tools import tool
from retriever import AdvancedRetriever, load_vector_store

# Initialize retriever globally (loaded once)
print("Loading vector store for agent tools...")
vector_store = load_vector_store()
retriever = AdvancedRetriever(vector_store)
print("✓ Agent tools ready")


@tool
def search_code(query: str) -> str:
    """
    Search for code snippets related to a query.
    Use this when user asks: "How does X work?", "Show me Y", "Find Z"
    
    Args:
        query: Natural language question about code
        
    Returns:
        Relevant code snippets with sources
    """
    print(f"[Tool: search_code] Query: {query}")
    
    # Use retriever to get relevant docs
    docs = retriever.retrieve(query, top_k=3)
    
    # Format results
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        results.append(f"**Source {i}: {source}**\n{doc.page_content}\n")
    
    return "\n---\n".join(results)


@tool
def explain_code(code_snippet: str) -> str:
    """
    Provide detailed explanation of a code snippet.
    Use this when user asks: "Explain this code", "What does this do?", "Break down X"
    
    Args:
        code_snippet: The code to explain (or description of code to find and explain)
        
    Returns:
        Detailed explanation
    """
    print(f"[Tool: explain_code] Snippet length: {len(code_snippet)} chars")
    
    # If snippet is short, it might be a search query
    if len(code_snippet) < 100:
        # Search for the code first
        docs = retriever.retrieve(code_snippet, top_k=2)
        if docs:
            code_snippet = docs[0].page_content
    
    # Return code with request for explanation
    return f"""Code to explain:
{code_snippet[:1000]}

(This will be explained by the LLM in the agent flow)"""


@tool  
def get_file_content(filename: str) -> str:
    """
    Retrieve the full content of a specific file.
    Use this when user asks: "Show me the full X file", "Get Y.py", "What's in Z?"
    
    Args:
        filename: Name of file to retrieve (e.g., "README.md", "auth.py")
        
    Returns:
        Full file content
    """
    print(f"[Tool: get_file_content] File: {filename}")
    
    # Get all documents
    all_data = vector_store.get()
    
    # Find chunks from this file
    file_chunks = []
    for i, metadata in enumerate(all_data['metadatas']):
        source = metadata.get('source', '')
        if filename.lower() in source.lower():
            file_chunks.append(all_data['documents'][i])
    
    if not file_chunks:
        return f"File '{filename}' not found in the indexed codebase."
    
    # Combine chunks
    full_content = "\n\n".join(file_chunks)
    
    return f"**File: {filename}**\n\n{full_content[:2000]}...\n\n(Showing first 2000 characters)"


# List of tools for agent
agent_tools = [search_code, explain_code, get_file_content]