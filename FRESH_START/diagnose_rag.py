import os
import logging
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose")

CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash-lite-001"

def diagnose():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set.")
        return

    logger.info("1. Initializing Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    
    logger.info("2. Connecting to ChromaDB...")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    logger.info("3. Testing Retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    query = "Tell me about Winston the dog."
    logger.info(f"Querying retriever with: '{query}'")
    try:
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents.")
        for i, doc in enumerate(docs):
            logger.info(f"Doc {i} length: {len(doc.page_content)}")
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        return

    logger.info("4. Setting up LLM and Chain...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a direct assistant. Answer based on context: {context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("5. Invoking full chain...")
    try:
        response = rag_chain.invoke({"input": query})
        logger.info("Chain response received!")
        print("-" * 30)
        print(response)
        print("-" * 30)
    except Exception as e:
        logger.error(f"Chain invocation failed: {e}")

if __name__ == "__main__":
    diagnose()
