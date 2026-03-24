from retriever import AdvancedRetriever, load_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

GEMINI_API_KEY = "key"
genai.configure(api_key=GEMINI_API_KEY)

def create_chat_session(vector_store):
    """Create chat with advanced retrieval"""
    
    # Initialize advanced retriever
    retriever = AdvancedRetriever(vector_store)
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )
    
    chat_history = []
    
    def chat(question):
        """Chat with advanced retrieval pipeline"""
        
        # Use advanced retrieval
        relevant_docs = retriever.retrieve(question, top_k=5)
        
        # Build context
        context_parts = []
        sources = []
        
        for doc in relevant_docs:
            context_parts.append(doc.page_content)
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build history
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"User: {q}\nAssistant: {a}" 
                for q, a in chat_history[-3:]
            ])
        
        # Prompt
        prompt = f"""You are a code documentation assistant.

Previous conversation:
{history_text if history_text else "None"}

Context from repository:
{context}

Question: {question}

Answer directly and specifically using the context. If asking about a file, summarize its actual content.

Answer:"""
        
        try:
            response = llm.invoke(prompt)
            answer = response.content
            
            chat_history.append((question, answer))
            
            output = f"{answer}\n\nSources: {', '.join([s.split('/')[-1] for s in sources[:3]])}"
            
            return output
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear():
        chat_history.clear()
        print("Chat cleared")
    
    return chat, clear

# ... rest of main() stays same

def main():
    """Interactive CLI"""
    
    print("="*60)
    print("GitHub Repository Assistant")
    print("="*60)
    print("Loading vector store...")
    
    vector_store = load_vector_store()
    chat, clear = create_chat_session(vector_store)
    
    print("✓ Ready! Type 'quit' to exit, 'clear' to reset conversation")
    print("="*60)
    print()
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'clear':
                clear()
                continue
            
            print("Assistant:", end=" ")
            response = chat(question)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()