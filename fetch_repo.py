from github import Github
import os

def fetch_repo_files(repo_name, file_extensions=['.py', '.js', '.md'], max_files=20):
    """
    Fetch files from a GitHub repository.
    
    Args:
        repo_name: "owner/repo" format (e.g., "facebook/react")
        file_extensions: List of extensions to fetch
        max_files: Limit number of files (for testing)
    
    Returns:
        List of documents with content and metadata
    """
    
    try:
        g = Github()  # No auth, 60 req/hour limit
        repo = g.get_repo(repo_name)
        
        print(f"Fetching files from {repo_name}...")
        print(f"Looking for: {', '.join(file_extensions)}")
        print("-" * 60)
        
        documents = []
        
        def explore_contents(contents, current_count):
            """Recursively explore repo contents"""
            
            for content in contents:
                # Stop if hit max_files
                if current_count >= max_files:
                    return current_count
                
                # If directory, recurse
                if content.type == "dir":
                    try:
                        current_count = explore_contents(
                            repo.get_contents(content.path),
                            current_count
                        )
                    except:
                        pass  # Skip inaccessible dirs
                
                # If file with correct extension
                elif content.type == "file":
                    if any(content.name.endswith(ext) for ext in file_extensions):
                        try:
                            # Fetch file content
                            file_content = content.decoded_content.decode('utf-8')
                            
                            # Store document
                            doc = {
                                'content': file_content,
                                'metadata': {
                                    'source': f"{repo_name}/{content.path}",
                                    'title': content.name,
                                    'path': content.path,
                                    'size': content.size,
                                    'url': content.html_url
                                }
                            }
                            
                            documents.append(doc)
                            current_count += 1
                            
                            print(f"✓ [{current_count}/{max_files}] {content.path}")
                            
                        except Exception as e:
                            print(f"✗ Failed: {content.path} - {str(e)}")
            
            return current_count
        
        # Start exploring from root
        root_contents = repo.get_contents("")
        explore_contents(root_contents, 0)
        
        print("-" * 60)
        print(f"✓ Fetched {len(documents)} files")
        
        return documents
        
    except Exception as e:
        print(f"✗ Error fetching repo: {str(e)}")
        return []

def preview_documents(documents):
    """Show preview of fetched documents"""
    
    print("\n" + "="*60)
    print("Document Preview")
    print("="*60)
    
    for i, doc in enumerate(documents[:3]):  # Show first 3
        print(f"\n[{i+1}] {doc['metadata']['title']}")
        print(f"Path: {doc['metadata']['path']}")
        print(f"Size: {doc['metadata']['size']} bytes")
        print("Content preview:")
        print("-" * 60)
        print(doc['content'][:200])
        print("...")
        print("-" * 60)

if __name__ == "__main__":
    # Test with a small, popular repo
    # Using 'requests' library (small Python library)
    repo_name = "psf/requests"
    
    print("="*60)
    print("Testing Repository File Fetching")
    print("="*60)
    print()
    
    # Fetch Python files only
    docs = fetch_repo_files(
        repo_name=repo_name,
        file_extensions=['.py', '.md'],
        max_files=10  # Just 10 for testing
    )
    
    if docs:
        preview_documents(docs)
        print(f"\n✓ Day 7 Part 1 Complete!")
        print(f"✓ Ready to integrate with ingest.py tomorrow")
    else:
        print("\n✗ No documents fetched")