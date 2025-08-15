import os, threading
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chatbot import Chatbot

app = FastAPI(title="Support Bot (not a crisis service)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Serve the simple UI
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

# ---- One global model; per-session we store just the history text
BOT: Optional[Chatbot] = None
BOT_LOCK = threading.Lock()
SESSIONS: Dict[str, Optional[str]] = {}  # session_id -> history_text

@app.on_event("startup")
def load_model_once():
    global BOT
    model_name = os.getenv("BOT_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"[startup] Loading model: {model_name}")
    BOT = Chatbot(model_name=model_name)
    BOT.reset_history()
    print("[startup] Model ready")

@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    if not payload.message or not payload.message.strip():
        raise HTTPException(400, "Empty message")
    if BOT is None:
        raise HTTPException(500, "Model not loaded")

    sid = payload.session_id or os.urandom(8).hex()
    if sid not in SESSIONS:
        SESSIONS[sid] = None

    try:
        with BOT_LOCK:  # serialize generation on small CPU plans
            BOT.history_text = SESSIONS[sid]          # restore this user's history
            reply = BOT.generate_reply(payload.message)
            SESSIONS[sid] = BOT.history_text          # save updated history
    except Exception as e:
        print("[/api/chat] error:", repr(e))
        raise HTTPException(500, "Model error")

    return ChatOut(reply=reply, session_id=sid)

@app.get("/healthz")
def healthz():
    return {"ok": True}
