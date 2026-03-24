from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def get_embeddings():
    return OllamaEmbeddings(
        model="mxbai-embed-large"
    )

def load_sample_documents():
    sample_docs = [
        {
            "content": """
            Deployment Process:
            1. Raise PR, get 2 approvals
            2. Run integration tests locally
            3. Merge to develop branch
            4. Jenkins pipeline triggers automatically
            5. Deploy to staging, run smoke tests
            6. Get QA sign-off
            7. Deploy to production Friday 2pm IST only
            8. Monitor 30 minutes post deployment
            """,
            "source": "confluence/deployment-process",
            "title": "Deployment Process"
        },
        {
            "content": """
            Kafka Configuration:
            Bootstrap servers: kafka-broker:9092
            Schema Registry URL: http://schema-registry:8081
            Consumer group naming: team.service.purpose
            Always use Schema Registry for new topics.
            Dead letter queues required for all consumers.
            Retention: 7 days standard, 30 days audit topics.
            """,
            "source": "confluence/kafka-setup",
            "title": "Kafka Configuration Guide"
        },
        {
            "content": """
            Onboarding Guide:
            Week 1: Laptop setup, VPN, VDI access
            Week 2: System architecture overview
            Week 3: First bug fix ticket
            Week 4: First feature development
            
            Contacts:
            - VPN issues: IT helpdesk
            - Access issues: Team lead
            - Architecture questions: Tech lead
            
            Daily standup: 10am IST
            Sprint planning: Every 2 weeks Monday
            """,
            "source": "confluence/onboarding",
            "title": "Developer Onboarding Guide"
        }
    ]
    
    return [
        Document(
            page_content=doc["content"],
            metadata={
                "source": doc["source"],
                "title": doc["title"]
            }
        )
        for doc in sample_docs
    ]

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Documents: {len(documents)} → Chunks: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """
    This is where the magic happens.
    Each chunk gets converted to a vector (array of numbers).
    Then stored in ChromaDB for semantic search.
    """
    print("\nGenerating embeddings... (this takes 10-15 seconds)")
    
    embeddings = get_embeddings()
    
    # This calls Ollama locally to convert text → numbers
    # Then stores in ./chroma_db folder
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")
    return vector_store

def test_semantic_search(vector_store):
    """
    Test that semantic search actually works.
    Notice: we search for "production deployment" 
    but it finds "Deploy to production Friday"
    That's semantic similarity, not keyword matching.
    """
    print("\n" + "="*50)
    print("Testing Semantic Search")
    print("="*50)
    
    test_queries = [
        "How do I deploy to production?",
        "Who should I contact for VPN problems?",
        "What's the Kafka retention period?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # This converts query to vector, finds closest chunks
        results = vector_store.similarity_search(query, k=2)
        
        print(f"Found in: {results[0].metadata['title']}")
        print(f"Content: {results[0].page_content[:100]}...")

    # Add to test_semantic_search function, after the loop
    print("\n" + "="*50)
    print("Experiment: Similar meanings, different words")
    print("="*50)

    similar_queries = [
        "deployment process",
        "how to push to prod",
        "production release steps"
    ]

    for query in similar_queries:
        results = vector_store.similarity_search(query, k=1)
        print(f"\nQuery: '{query}'")
        print(f"Found: {results[0].metadata['title']}")




if __name__ == "__main__":
    docs = load_sample_documents()
    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)
    test_semantic_search(vector_store)