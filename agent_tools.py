"""
Tools for LangGraph agent
"""

from langchain.tools import tool
from retriever import AdvancedRetriever, load_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import SecretStr


# Initialize retriever globally (loaded once)
print("Loading vector store for agent tools...")
vector_store = load_vector_store()
retriever = AdvancedRetriever(vector_store)
print("Agent tools ready")

explain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=SecretStr(os.getenv("GEMINI_API_KEY", ""))
)

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
    prompt = f"""Explain this code clearly and concisely for a developer trying to understand it:
        {code_snippet[:1500]}
        Explain:
        1. What it does
        2. Key logic/algorithm used
        3. Any notable patterns or gotchas"""
    
    response = explain_llm.invoke([prompt])
    content = response.content
    if isinstance(content, list):
        content = " ".join([str(msg) for msg in content])
    return content


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