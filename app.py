import os
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chatbot import Chatbot  # TinyLlama therapist-safe bot

app = FastAPI(title="Support Bot (not a crisis service)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Serve the simple UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    index_path = os.path.join("static", "index.html")
    return FileResponse(index_path)

# Simple per-session memory (in-memory). For production, use Redis.
SESSIONS: Dict[str, Chatbot] = {}

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None  # sticky sessions by ID

class ChatOut(BaseModel):
    reply: str
    session_id: str

@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message")

    sid = payload.session_id or os.urandom(8).hex()
    bot = SESSIONS.get(sid)
    if bot is None:
        bot = Chatbot(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        SESSIONS[sid] = bot

    try:
        reply = bot.generate_reply(msg)
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    return ChatOut(reply=reply, session_id=sid)

@app.get("/healthz")
def healthz():
    return {"ok": True}
