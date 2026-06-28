import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from github import Github
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
github_token = os.getenv("GITHUB_TOKEN")
print(github_token)

def fetch_repo_files(repo_name, max_files=20):
    """
    Fetch files from GitHub repository
    """
    print(f"Fetching from {repo_name}...")
    
    try:
        g = Github(github_token) if github_token else Github()
        repo = g.get_repo(repo_name)
        
        documents = []
        file_count = 0
        
        def explore(contents):
            nonlocal file_count
            
            for content in contents:
                if file_count >= max_files:
                    return
                # SKIP certain directories
                skip_dirs = ['.github', 'tests', 'docs/_themes', '.git']
                if any(skip in content.path for skip in skip_dirs):
                    continue
                if content.type == "dir":
                    try:
                        explore(repo.get_contents(content.path))
                    except:
                        pass
                
                elif content.type == "file":
                    # PRIORITIZE .py files in src/
                    if content.name.endswith(('.py', '.md')):
                        # Prefer files in src/ directory
                        priority = 0 if 'src/' in content.path else 1
                    # Only Python and Markdown files
                    if content.name.endswith(('.py', '.md')):
                        try:
                            file_content = content.decoded_content.decode('utf-8')
                            
                            doc = Document(
                                page_content=file_content,
                                metadata={
                                    'source': f"{repo_name}/{content.path}",
                                    'title': content.name,
                                    'path': content.path,
                                    'repo': repo_name,
                                    'url': content.html_url,
                                    'priority': priority  # For sorting later
                                }
                            )
                            
                            documents.append(doc)
                            file_count += 1
                            print(f"  ✓ [{file_count}/{max_files}] {content.path}")
                            
                        except Exception as e:
                            pass
        
        # Start exploring
        explore(repo.get_contents(""))
        # Sort by priority (src/ files first)
        documents.sort(key=lambda x: x.metadata.get('priority', 1))
        print(f"\n✓ Fetched {len(documents)} files")
        return documents
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def chunk_documents(documents):
    """
    Split documents into chunks
    For code: smaller chunks work better
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger for code files
        chunk_overlap=150,
        separators=[
            "\n\nclass ",    # Prioritize class boundaries
            "\n\ndef ",      # Then function boundaries
            "\n\n",          # Then double newlines
            "\n",            # Then single newlines
            " "              # Finally spaces
        ]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Documents: {len(documents)} → Chunks: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """
    Generate embeddings and store in ChromaDB
    """
    print("\nGenerating embeddings (this takes 30-60 seconds)...")
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Delete old vector store if exists
    import shutil
    try:
        shutil.rmtree("./chroma_db")
        print("✓ Cleared old vector store")
    except:
        pass
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")
    return vector_store

def test_search(vector_store):
    """
    Test that semantic search works
    """
    print("\n" + "="*60)
    print("Testing Search")
    print("="*60)
    
    test_queries = [
        "How do I make an HTTP request?",
        "What are the main classes?",
        "Show me authentication code"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=2)
        if results:
            print(f"  Found in: {results[0].metadata['path']}")
            print(f"  Content preview: {results[0].page_content[:100]}...")

if __name__ == "__main__":
    print("="*60)
    print("Building GitHub Repository Knowledge Base")
    print("="*60)
    print()
    
    # Use a small, well-documented repo
    # 'requests' library - popular Python HTTP library
    repo_name = "psf/requests"
    
    # Step 1: Fetch files
    docs = fetch_repo_files(repo_name, max_files=15)
    
    if not docs:
        print("No documents fetched. Exiting.")
        exit()
    
    # Step 2: Chunk
    chunks = chunk_documents(docs)
    
    # Step 3: Create vector store
    vector_store = create_vector_store(chunks)
    
    # Step 4: Test
    test_search(vector_store)
    
    print("\n" + "="*60)
    print("✓ Day 7 Complete!")
    print("✓ ChromaDB created with GitHub repo data")
    print("✓ Ready to use with chat.py tomorrow")
    print("="*60)