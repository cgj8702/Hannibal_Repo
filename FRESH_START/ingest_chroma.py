"""
ChromaDB Ingest Script - Debug Version
Creates a ChromaDB collection from the same screenplay data used for FAISS

WARNING: ChromaDB 1.4.x has a known bug on Windows where add() silently crashes.
This script will likely fail on Windows. Use ingest.py (FAISS) instead.
See chroma_debug.py for full details on the issue and workarounds.
"""
import os
import re
import glob
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

# --- CONFIGURATION ---
CLEAN_SCRIPTS_DIR = "_archive/clean_scripts"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004"
COLLECTION_NAME = "hannibal_screenplays"


def extract_episode_id(filename: str) -> str:
    """Extract episode identifier from filename (e.g., 'CLEAN_Hannibal_1x01_Aperitif.txt' -> '1x01')"""
    m = re.search(r"(\d+x\d+)", filename)
    return m.group(1) if m else ""


def load_screenplays(directory):
    """Reads all .txt files from the directory."""
    scripts = []
    files = glob.glob(os.path.join(directory, "*.txt"))
    print(f"Found {len(files)} clean scripts.")
    
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(filepath)
            scripts.append((filename, content))
            
    return scripts


def chunk_by_scene(text, filename):
    """Splits text by Scene Headers (INT. / EXT.)."""
    scene_pattern = re.compile(r'(?=\n(?:INT\.|EXT\.))')
    chunks = scene_pattern.split(text)
    episode_id = extract_episode_id(filename)
    
    documents = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        first_line = chunk.strip().split('\n')[0]
        
        documents.append({
            "content": chunk.strip(),
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "scene_header": first_line[:100],
                "episode": episode_id
            }
        })
        
    return documents


def ingest_to_chroma():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set.")
        return False

    print("--- Starting ChromaDB Ingestion ---")
    
    # 1. Load Scripts
    scripts = load_screenplays(CLEAN_SCRIPTS_DIR)
    if not scripts:
        print(f"No scripts found in {CLEAN_SCRIPTS_DIR}.")
        return False

    # 2. Chunk all scripts
    all_documents = []
    print("Chunking scripts by scene...")
    for filename, content in scripts:
        docs = chunk_by_scene(content, filename)
        all_documents.extend(docs)
        print(f"  -> {filename}: {len(docs)} scenes")
        
    print(f"Total chunks created: {len(all_documents)}")
    
    # 3. Initialize ChromaDB with persistent storage
    print(f"\nInitializing ChromaDB at '{CHROMA_DB_DIR}'...")
    
    # Delete existing DB if present
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        shutil.rmtree(CHROMA_DB_DIR)
        print("  Removed existing ChromaDB directory.")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # 4. Create collection
    print(f"Creating collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Hannibal TV series screenplays"}
    )
    
    # 5. Create embeddings using Google's embedding model
    print("Initializing Google embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    
    # 6. Add documents in batches (ChromaDB has batch limits)
    batch_size = 100
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    
    print(f"\nAdding {len(all_documents)} documents in {total_batches} batches...")
    
    for i in tqdm(range(0, len(all_documents), batch_size), desc="Embedding"):
        batch = all_documents[i:i + batch_size]
        
        # Extract content and metadata
        texts = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        
        # Generate embeddings for this batch
        try:
            batch_embeddings = embeddings.embed_documents(texts)
        except Exception as e:
            print(f"\nError generating embeddings for batch {i//batch_size}: {e}")
            continue
        
        # Add to ChromaDB
        try:
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=batch_embeddings
            )
        except Exception as e:
            print(f"\nError adding batch {i//batch_size} to ChromaDB: {e}")
            continue
    
    # 7. Verify
    count = collection.count()
    print(f"\n--- Ingestion Complete! ---")
    print(f"ChromaDB collection '{COLLECTION_NAME}' now has {count} documents.")
    print(f"Database saved to: {os.path.abspath(CHROMA_DB_DIR)}")
    
    return True


if __name__ == "__main__":
    success = ingest_to_chroma()
    if not success:
        print("\nIngestion failed!")
        exit(1)
