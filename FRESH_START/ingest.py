import os
import re
import glob
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Helper to extract episode identifier from filename (e.g., "CLEAN_Hannibal_1x01_Aperitif.txt" -> "1x01")
def extract_episode_id(filename: str) -> str:
    m = re.search(r"(\d+x\d+)", filename)
    return m.group(1) if m else ""

# --- CONFIGURATION ---
CLEAN_SCRIPTS_DIR = "clean_scripts"
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/text-embedding-004" # Upgraded to newer model 

def load_screenplays(directory):
    """
    Reads all .txt files from the directory.
    Returns a list of (filename, content) tuples.
    """
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
    """
    Splits text by Scene Headers (INT. / EXT.).
    Returns a list of LangChain Document objects with metadata.
    """
    # Regex to find scene headers (e.g., "INT. KITCHEN - DAY")
    # We look for INT. or EXT. at the start of a line
    scene_pattern = re.compile(r'(?=\n(?:INT\.|EXT\.))')
    
    # Split text
    chunks = scene_pattern.split(text)
    
    # Extract episode ID once for all chunks from this file
    episode_id = extract_episode_id(filename)
    
    documents = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        # Try to extract the scene header for metadata
        first_line = chunk.strip().split('\n')[0]
        
        doc = Document(
            page_content=chunk.strip(),
            metadata={
                "source": filename,
                "chunk_index": i,
                "scene_header": first_line[:100],  # truncate for safety
                "episode": episode_id  # Add episode for chronological sorting
            }
        )
        documents.append(doc)
        
    return documents

def ingest_data():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set.")
        return

    print("--- Starting Ingestion ---")
    
    # 1. Load Scripts
    scripts = load_screenplays(CLEAN_SCRIPTS_DIR)
    if not scripts:
        print(f"No scripts found in {CLEAN_SCRIPTS_DIR}. Run auto_editor.py first.")
        return

    # 2. Chunking
    all_documents = []
    print("Chunking scripts by scene...")
    for filename, content in scripts:
        docs = chunk_by_scene(content, filename)
        all_documents.extend(docs)
        print(f"  -> {filename}: {len(docs)} scenes")
        
    print(f"Total chunks created: {len(all_documents)}")
    
    # 3. Create Vector Store
    print(f"Creating FAISS index from {len(all_documents)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    
    # Simple direct creation - FAISS handles the whole list well
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    # Save locally as requested
    print(f"Saving FAISS index to '{FAISS_INDEX_DIR}'...")
    vectorstore.save_local(FAISS_INDEX_DIR)
        
    print("--- Ingestion Complete! ---")
    print(f"Database saved to: {os.path.abspath(FAISS_INDEX_DIR)}")

if __name__ == "__main__":
    ingest_data()
