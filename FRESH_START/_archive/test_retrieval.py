import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_retrieval")

FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/text-embedding-004"

def test():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found.")
        return

    logger.info("Initializing Embeddings...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        if not os.path.exists(FAISS_INDEX_DIR):
            logger.error(f"Vector DB not found at {FAISS_INDEX_DIR}")
            return

        logger.info("Loading FAISS index...")
        vs = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        
        query = "Tell me about Hannibal's social exclusion paper"
        logger.info(f"Testing retrieval for query: '{query}'")
        
        docs = retriever.invoke(query)
        
        logger.info(f"Retrieved {len(docs)} documents.")
        for i, doc in enumerate(docs):
            ep = doc.metadata.get("episode", "N/A")
            hdr = doc.metadata.get("scene_header", "No Header")
            logger.info(f"  [{i}] Episode: {ep} | Scene: {hdr}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

if __name__ == "__main__":
    test()
