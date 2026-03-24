from github import Github

def test_github_connection():
    """Test GitHub API - no auth needed for public repos"""
    
    try:
        # Connect (no auth, rate limited to 60 requests/hour)
        g = Github()
        
        # Test with React repo
        repo = g.get_repo("facebook/react")
        
        print("GitHub API working!")
        print(f"\nRepo: {repo.full_name}")
        print(f"Stars: {repo.stargazers_count}")
        
        # Fetch README
        readme = repo.get_contents("README.md")
        content = readme.decoded_content.decode('utf-8')
        
        print(f"\nFetched README ({len(content)} chars)")
        print("First 300 chars:")
        print("-" * 60)
        print(content[:300])
        print("-" * 60)
        
        # List some files
        print("\nListing source files:")
        contents = repo.get_contents("packages/react/src")
        
        count = 0
        for item in contents:
            if item.type == "file" and item.name.endswith('.js'):
                print(f"  - {item.path} ({item.size} bytes)")
                count += 1
                if count >= 5:  # Just show 5 files
                    break
        
        print("\nDay 6 Complete! GitHub API working.")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing GitHub API...")
    print("="*60)
    test_github_connection()