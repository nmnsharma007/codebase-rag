from retriever import load_vector_store

vector_store = load_vector_store()

# Get all data
all_data = vector_store.get()

# Find README chunks
for i, source in enumerate(all_data['metadatas']):
    done = False
    if 'README' in source.get('source', ''):
        print(f"\n=== README Chunk {i} ===")
        print(f"Source: {source.get('source')}")
        print(f"Content preview:")
        print(all_data['documents'][i][:300])
        print("="*60)
        done = True
        break
    if done:
        break