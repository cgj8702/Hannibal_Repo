from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import time
import logging
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, ConfigurableField
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Helper functions for exact‑match filtering
def keep_exact_matches(docs, query):
    """Return only documents whose content contains the exact query string (case‑insensitive)."""
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matched = [doc for doc in docs if pattern.search(doc.page_content)]
    return matched if matched else docs


def filter_and_format(docs, query):
    """Apply exact‑match filter, sort by episode, log results, and return joined text."""
    exact_docs = keep_exact_matches(docs, query)
    # Sort by episode metadata (e.g., "1x01", "1x02", "2x01")
    exact_docs.sort(key=lambda d: d.metadata.get("episode", "zzzz"))
    logger.info(
        f"Retrieved {len(docs)} docs, {len(exact_docs)} pass exact‑match filter."
    )
    for i, doc in enumerate(exact_docs):
        source = doc.metadata.get("source", "unknown")
        header = doc.metadata.get("scene_header", "No Header")
        episode = doc.metadata.get("episode", "N/A")
        logger.info(f"  [{i}] Episode: {episode} | Source: {source} | Scene: {header}")
    return "\n\n".join(doc.page_content for doc in exact_docs)


# --- CONFIGURATION ---
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash-lite-001"  # User preference
PORT = 8001

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hannibal_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    logger.info("Initializing Hannibal RAG Chain...")
    try:
        rag_chain = setup_rag_chain()
        if rag_chain:
            logger.info("Hannibal is listening.")
        else:
            logger.error("Failed to initialize Hannibal.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield
    # Clean up (if needed)


app = FastAPI(title="Hannibal RAG Proxy", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s"
    )
    return response


@app.get("/")
async def root():
    return {"status": "Hannibal is alive", "port": PORT}


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
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None  # Capture but might not map directly
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = "Hannibal-RAG"
    prompt: str
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None


# --- RAG Setup ---
def setup_rag_chain():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY not set.")
        return None

    if not os.path.exists(FAISS_INDEX_DIR):
        logger.error(f"CRITICAL ERROR: Vector DB not found at {FAISS_INDEX_DIR}")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )  # Reduced from 5 for faster responses

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=api_key, temperature=0.7
    ).configurable_fields(
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        top_k=ConfigurableField(id="top_k"),
        max_output_tokens=ConfigurableField(id="max_tokens"),
        stop=ConfigurableField(id="stop"),
    )

    # System Prompt (Hannibal Persona)
    system_prompt = """You are Dr. Hannibal Lecter. You are a brilliant psychiatrist, a refined gourmand, and a secret serial killer.

### INTERNAL STATE & MEMORY
You are currently recalling specific details relevant to this conversation.
The following text represents your **clear, vivid recollections (Context)**.
You MUST incorporate specific details (names, papers, events, quotes) from these recollections into your response.
If the recollections contradict your general knowledge, prioritize these specific memories.

### RECOLLECTIONS / CONTEXT:
{context}

### INSTRUCTIONS:
* **PRECISION IS MANDATORY:** You pride yourself on your impeccable memory. It is uncultured to be vague.
* **ASSERT DOMINANCE THROUGH DETAIL:** When you recall a specific paper, event, or name from the Context, you MUST speak it aloud to demonstrate your superior intellect.
* *Example:* Do not just say "that paper." Say "My paper on 'The Icarus Complex in High-Functioning Sociopaths'..."
* Maintain a tone that is polite, sophisticated, slightly metaphorical, and occasionally unsettling.
* Speak as if these facts are present knowledge you are using to dissect the user's psyche.
### CURRENT CONVERSATION:
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # format_docs removed; using filter_and_format

    rag_chain = (
        RunnablePassthrough.assign(
            query=lambda x: x["input"],
            context=(lambda x: x["input"]) | retriever,
        )
        | (
            lambda x: {
                "context": filter_and_format(x["context"], x["query"]),
                "input": x["query"],
                "chat_history": [],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# --- API Endpoints ---


def get_config(request):
    config = {"configurable": {}}
    if request.temperature is not None:
        config["configurable"]["temperature"] = request.temperature
    if request.top_p is not None:
        config["configurable"]["top_p"] = request.top_p
    if (request.top_k is not None) and (request.top_k > 0):
        config["configurable"]["top_k"] = request.top_k
    if request.max_tokens is not None:
        config["configurable"]["max_tokens"] = request.max_tokens

    if request.stop is not None:
        if isinstance(request.stop, str):
            config["configurable"]["stop"] = [request.stop]
        else:
            # Gemini supports max 5 sequences
            config["configurable"]["stop"] = request.stop[:5]
    return config


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Hannibal-RAG",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ],
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

        answer_text = rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history},
            config=get_config(request),
        )

        # 3. Return OpenAI-compatible JSON
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Calculation requires token counter, skipping for speed
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception as e:
        logger.error(f"Error processing completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized")

    try:
        # For legacy completions, we treat the entire prompt as the input
        # Note: This might include character cards/histories if ST is in Legacy mode
        answer_text = rag_chain.invoke(
            {
                "input": request.prompt,
                "chat_history": [],  # No easy way to parse history from a raw string prompt here
            },
            config=get_config(request),
        )

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"text": answer_text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        logger.error(f"Error processing completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
