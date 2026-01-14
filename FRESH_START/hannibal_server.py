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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.runnables import (
    RunnablePassthrough,
    ConfigurableField,
    ensure_config,
    RunnableConfig,
)
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
    """Sort by episode, log results, and return joined text."""
    # Sort by episode metadata (e.g., "1x01", "1x02", "2x01")
    docs.sort(key=lambda d: d.metadata.get("episode", "zzzz"))
    logger.info(f"Retrieved {len(docs)} docs for context.")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        header = doc.metadata.get("scene_header", "No Header")
        episode = doc.metadata.get("episode", "N/A")
        logger.info(f"  [{i}] Episode: {episode} | Source: {source} | Scene: {header}")
    return "\n\n".join(doc.page_content for doc in docs)


# --- CONFIGURATION ---
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash-lite"  # Best available cost-efficient model
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
retriever = None
vectorstore = None


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
    temperature: Optional[float] = None
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

    # The Google API key is read from the environment by the client library;
    # avoid passing an unknown keyword argument into the embeddings constructor.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vs = FAISS.load_local(
        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    # expose vectorstore for diagnostics
    global vectorstore
    vectorstore = vs
    retriever_local = vs.as_retriever(
        search_kwargs={"k": 5}
    )  # Increased to 5 to match retrieval depth used in tests
    # expose retriever for diagnostics/logging
    global retriever
    retriever = retriever_local

    # Construct LLM runnable. Authentication is handled via environment vars,
    # so do not pass `google_api_key` here (not a valid ctor arg).
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ],
    ).configurable_fields(
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        top_k=ConfigurableField(id="top_k"),
        max_output_tokens=ConfigurableField(id="max_tokens"),
        stop=ConfigurableField(id="stop"),
    )

    # System Prompt - minimal, just injects RAG context
    # SillyTavern handles the persona/character card
    system_prompt = """The following context contains relevant information retrieved from your knowledge base. Use it to inform your responses:

{context}"""

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
            context=(lambda x: x["input"]) | retriever_local,
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


def get_config(request) -> RunnableConfig:
    # Build a RunnableConfig dict explicitly so it matches the expected
    # TypedDict shape used by the `Runnable` APIs.
    config: RunnableConfig = {
        "configurable": {},
        "tags": [],
        "metadata": {},
    }
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
    # Return an ensured RunnableConfig (fills defaults)
    return ensure_config(config)


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
        user_input = request.messages[-1].content
        chat_history = []
        for msg in request.messages[:-1]:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))

        # Diagnostic retrieval/logging
        try:
            docs = None
            if vectorstore is not None:
                try:
                    docs = vectorstore.similarity_search(user_input, k=5)
                except Exception:
                    docs = None
            if (
                docs is None
                and retriever is not None
                and hasattr(retriever, "get_relevant_documents")
            ):
                method = getattr(retriever, "get_relevant_documents", None)
                if callable(method):
                    docs = method(user_input)
            if docs:
                logger.info(
                    f"Diagnostic retrieval: returned {len(docs)} docs for query"
                )
                for i, d in enumerate(docs):
                    logger.info(
                        f"  [diag {i}] Episode: {d.metadata.get('episode','N/A')} | Source: {d.metadata.get('source','unknown')} | Scene: {d.metadata.get('scene_header','No Header')}"
                    )
                try:
                    ctx_snip = " | ".join(
                        f"{d.metadata.get('episode','N/A')}:{d.metadata.get('scene_header','No Header')}"
                        for d in docs[:5]
                    )
                    logger.info(f"Diagnostic context snippet: {ctx_snip}")
                except Exception:
                    pass
            else:
                logger.warning("Diagnostic retrieval returned no docs or failed")
        except Exception as e:
            logger.warning(f"Diagnostic retrieval failed: {e}")

        # Invoke the chain
        answer_text = rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history},
            config=get_config(request),
        )

        logger.info(f"Model response (chat): {answer_text}")

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
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception as e:
        logger.error(f"Error processing chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized")

    try:
        # Diagnostic retrieval/logging
        try:
            docs = None
            if vectorstore is not None:
                try:
                    docs = vectorstore.similarity_search(request.prompt, k=5)
                except Exception:
                    docs = None
            if (
                docs is None
                and retriever is not None
                and hasattr(retriever, "get_relevant_documents")
            ):
                method = getattr(retriever, "get_relevant_documents", None)
                if callable(method):
                    docs = method(request.prompt)
            if docs:
                logger.info(
                    f"Diagnostic retrieval: returned {len(docs)} docs for prompt"
                )
                for i, d in enumerate(docs):
                    logger.info(
                        f"  [diag {i}] Episode: {d.metadata.get('episode','N/A')} | Source: {d.metadata.get('source','unknown')} | Scene: {d.metadata.get('scene_header','No Header')}"
                    )
            else:
                logger.warning("Diagnostic retrieval returned no docs or failed")
        except Exception as e:
            logger.warning(f"Diagnostic retrieval failed: {e}")

        answer_text = rag_chain.invoke(
            {"input": request.prompt, "chat_history": []}, config=get_config(request)
        )

        logger.info(f"Model response (completion): {answer_text}")

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


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized")

    try:
        # For legacy completions, we treat the entire prompt as the input
        # Note: This might include character cards/histories if ST is in Legacy mode
        # For legacy completions we pass a RunnableConfig constructed from
        # the request to allow configurable fields to apply.
        # Diagnostic retrieval/logging similar to chat endpoint
        docs = None
        try:
            if retriever:
                method = getattr(retriever, "get_relevant_documents", None)
                if callable(method):
                    docs = method(request.prompt)
                    if isinstance(docs, (list, tuple)):
                        logger.info(
                            f"Diagnostic retrieval: returned {len(docs)} docs for prompt"
                        )
                        for i, d in enumerate(docs):
                            logger.info(
                                f"  [diag {i}] Episode: {d.metadata.get('episode','N/A')} | Source: {d.metadata.get('source','unknown')} | Scene: {d.metadata.get('scene_header','No Header')}"
                            )
                    else:
                        logger.warning(
                            "Diagnostic retrieval returned non-sequence result; skipping detailed logging"
                        )
        except Exception as e:
            logger.warning(f"Diagnostic retrieval failed: {e}")

        answer_text = rag_chain.invoke(
            {"input": request.prompt, "chat_history": []}, config=get_config(request)
        )

        logger.info(f"Model response (completion): {answer_text}")

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
