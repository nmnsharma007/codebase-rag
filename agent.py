"""
LangGraph agent orchestration
"""

import os

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pydantic import SecretStr
from agent_tools import agent_tools

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def create_code_agent():
    """
    Create LangGraph agent with code tools
    """
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=SecretStr(os.getenv("GEMINI_API_KEY", ""))
    )
    
    # System prompt for agent
    system_prompt = """You are a helpful code assistant with access to tools for searching and analyzing code.

        When a user asks a question:
        1. Decide which tool(s) to use
        2. Call the appropriate tool(s)  
        3. Provide a clear, helpful answer based on the tool results

        Available tools:
        - search_code: Find relevant code snippets
        - explain_code: Get detailed explanation of code
        - get_file_content: Retrieve full file content

        Be concise and cite sources when providing answers."""
    
    # Create agent
    agent = create_react_agent(
        llm,
        agent_tools,
        prompt=system_prompt
    )
    
    return agent


def chat_with_agent(agent, query):
    """
    Send query to agent and get response
    """
    print(f"\n[Agent] Processing: {query}")
    print("-" * 60)
    
    # Invoke agent
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract final response
    messages = result.get("messages", [])
    
    # Get last message (agent's final response)
    if messages:
        final_message = messages[-1]
        response = final_message.content
        return response
    
    return "No response generated"


def main():
    """Interactive agent chat"""
    
    print("="*60)
    print("Code Assistant Agent (powered by LangGraph)")
    print("="*60)
    print("Initializing agent...")
    
    agent = create_code_agent()
    
    print("✓ Agent ready!")
    print("\nAvailable commands:")
    print("  - Ask any question about code")
    print("  - 'quit' to exit")
    print("="*60)
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Get response from agent
            response = chat_with_agent(agent, query)
            
            print(f"\nAgent: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()