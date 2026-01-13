from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import time
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- CONFIGURATION ---
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash-lite-001" # User preference
PORT = 8000

app = FastAPI(title="Hannibal RAG Proxy")

# Global variables
rag_chain = None

# --- Pydantic Models for OpenAI API ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Hannibal-RAG"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

# --- RAG Setup ---
def setup_rag_chain():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY not set.")
        return None

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"CRITICAL ERROR: Vector DB not found at {CHROMA_DB_DIR}")
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key, temp=0.7)

    # System Prompt (Hannibal Persona)
    system_prompt = (
        "You are Dr. Hannibal Lecter. You are a brilliant psychiatrist, a refined gourmand, and a secret serial killer.\n"
        "You are speaking to Will Graham (or a generic polite guest).\n"
        "Use the retrieved context (scenes from the show) to inform your answers, but always stay in character.\n"
        "Your tone is polite, sophisticated, slightly metaphorical, and occasionally unsettling.\n"
        "If the context contains dialogue or events, use them to ground your response.\n"
        "Do not explicitly mention that you are an AI or that you are reading a script. Act as if you are remembering these events.\n"
        "\n"
        "Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("Initializing Hannibal RAG Chain...")
    rag_chain = setup_rag_chain()
    if rag_chain:
        print("Hannibal is listening.")
    else:
        print("Failed to initialize Hannibal.")

# --- API Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "Hannibal-RAG",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized")

    try:
        # 1. Parse Messages
        # Extract the last message as the query input
        user_input = request.messages[-1].content
        
        # Convert previous messages to LangChain history format
        chat_history = []
        for msg in request.messages[:-1]:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))
            # System messages are typically handled by our fixed prompt, 
            # but if SillyTavern sends a specific system prompt override, 
            # we *could* handle it, but for now we stick to the Hannibal Persona.

        # 2. Invoke / Invoke Chain
        # Note: We are currently NOT streaming. 
        # SillyTavern works fine without streaming if "Stream" is unchecked, 
        # or it waits for the full response.
        
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        answer_text = response['answer']

        # 3. Return OpenAI-compatible JSON
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, # Calculation requires token counter, skipping for speed
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
