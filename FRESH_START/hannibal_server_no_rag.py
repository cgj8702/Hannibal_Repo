from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uvicorn
import os
import time
import logging
from contextlib import asynccontextmanager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (
    ConfigurableField,
    ensure_config,
    RunnableConfig,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# --- CONFIGURATION ---
LLM_MODEL = "gemini-2.5-flash-lite"
PORT = 8002

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hannibal_server_no_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_chain
    logger.info("Initializing Hannibal (No-RAG)...")
    try:
        llm_chain = setup_chain()
        if llm_chain:
            logger.info("Hannibal is listening on port 8002.")
        else:
            logger.error("Failed to initialize Hannibal.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield


app = FastAPI(title="Hannibal No-RAG Proxy", lifespan=lifespan)


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
    return {"status": "Hannibal is alive (No-RAG)", "port": PORT}


# Global variable
llm_chain = None


# --- Pydantic Models for OpenAI API ---
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Hannibal-No-RAG"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = "Hannibal-No-RAG"
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


# --- Chain Setup ---
def setup_chain():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("CRITICAL ERROR: GOOGLE_API_KEY not set.")
        return None

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL).configurable_fields(
        temperature=ConfigurableField(id="temperature"),
        top_p=ConfigurableField(id="top_p"),
        top_k=ConfigurableField(id="top_k"),
        max_output_tokens=ConfigurableField(id="max_tokens"),
        stop=ConfigurableField(id="stop"),
    )

    # Simplified prompt template for Direct LLM access
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain


# --- API Endpoints ---

def get_config(request) -> RunnableConfig:
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
            config["configurable"]["stop"] = request.stop[:5]
    return ensure_config(config)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Hannibal-No-RAG",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global llm_chain
    if not llm_chain:
        raise HTTPException(status_code=500, detail="Chain not initialized")

    try:
        user_input = request.messages[-1].content
        chat_history = []
        for msg in request.messages[:-1]:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                chat_history.append(SystemMessage(content=msg.content))

        answer_text = llm_chain.invoke(
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
    global llm_chain
    if not llm_chain:
        raise HTTPException(status_code=500, detail="Chain not initialized")

    try:
        answer_text = llm_chain.invoke(
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
