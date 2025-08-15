import os, threading
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from chatbot import Chatbot

app = FastAPI(title="Support Bot (not a crisis service)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Serve UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    session_id: str

# ---- One global model; sessions keep only text history ----
BOT: Optional[Chatbot] = None
BOT_LOCK = threading.Lock()
SESSIONS: Dict[str, Optional[str]] = {}  # session_id -> history_text

@app.on_event("startup")
def load_model_once():
    global BOT
    print("[startup] Loading GGUF model (llama.cpp backend)")
    BOT = Chatbot()  # reads MODEL_PATH/MODEL_URL env if present
    BOT.reset_history()
    print("[startup] Model ready")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    if not payload.message or not payload.message.strip():
        raise HTTPException(400, "Empty message")
    if BOT is None:
        raise HTTPException(500, "Model not loaded")

    sid = payload.session_id or os.urandom(8).hex()
    if sid not in SESSIONS:
        SESSIONS[sid] = None

    with BOT_LOCK:  # serialize generation for tiny CPU plans
        BOT.history_text = SESSIONS[sid]
        reply = BOT.generate_reply(payload.message)
        SESSIONS[sid] = BOT.history_text

    return ChatOut(reply=reply, session_id=sid)

@app.post("/api/chat_stream")
def chat_stream(payload: ChatIn):
    if not payload.message or not payload.message.strip():
        raise HTTPException(400, "Empty message")
    if BOT is None:
        raise HTTPException(500, "Model not loaded")

    sid = payload.session_id or os.urandom(8).hex()
    if sid not in SESSIONS:
        SESSIONS[sid] = None

    def gen():
        with BOT_LOCK:
            BOT.history_text = SESSIONS[sid]
            partial = ""
            for chunk in BOT.stream_reply(payload.message):
                partial += chunk
                yield chunk
            SESSIONS[sid] = BOT.history_text

    return StreamingResponse(gen(), media_type="text/plain")
