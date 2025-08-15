from typing import Optional, Generator
import os
import re
import pathlib
import requests

# ===== Safety & resources =====

UK_RESOURCES = (
    "• Emergency (UK): Call 999 now if you are in immediate danger.\n"
    "• Samaritans: 116 123 (free, 24/7) or jo@samaritans.org\n"
    "• Shout (text): 85258\n"
    "• NHS 111: urgent medical help when it’s not life-threatening"
)

THERAPY_SYSTEM = (
    "<|system|>\n"
    "You are a supportive, non-judgmental assistant. You are NOT a therapist and do NOT diagnose.\n"
    "Goals: listen briefly, reflect feelings, ask gentle open questions, offer coping tips and resources on request.\n"
    "If a message indicates self-harm, suicide, abuse, or immediate danger, STOP and ask the user to seek urgent help; "
    "encourage contacting emergency services and crisis lines. Do not give means-specific advice.\n"
    "Style: warm, concise, trauma-informed, respect autonomy, avoid medical claims, invite options.\n"
    "<|end|>\n"
)

CRISIS_PATTERNS = [
    r"\b(kill myself|suicide|end my life|want to die|don['’]t want to live)\b",
    r"\b(self[- ]?harm|cutting|overdose|od|take all my pills)\b",
    r"\b(hurt (myself|my self))\b",
    r"\b(abuse|assault|raped?|molest(ed)?)\b",
    r"\b(immediate danger|can'?t keep myself safe)\b",
]
_CRISIS_RE = [re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS]

def safety_check(text: str) -> str:
    t = text or ""
    for rx in _CRISIS_RE:
        if rx.search(t):
            return "crisis"
    return "ok"

def crisis_reply() -> str:
    return (
        "I’m really sorry you’re feeling this way. I’m not a crisis service, but I want you to be safe.\n\n"
        "If you are in immediate danger or feel unable to keep yourself safe, please call 999 now.\n\n"
        f"{UK_RESOURCES}\n\n"
        "If you’d like, I can stay with you here while you reach out. Would you like some grounding tips?"
    )

_HELPLINE_PATTERNS = [
    r"\b(who.*call.*mental\s*health)\b",
    r"\b(mental\s*health (helpline|hotline|number))\b",
    r"\b(crisis (line|helpline|number))\b",
    r"\b(who.*do i call)\b",
    r"\b(need.*help.*(call|number))\b",
]
_HELPLINE_RE = [re.compile(p, re.IGNORECASE) for p in _HELPLINE_PATTERNS]

def helpline_intent(text: str) -> bool:
    t = text or ""
    return any(rx.search(t) for rx in _HELPLINE_RE)

def helpline_reply() -> str:
    return (
        "If you’re in the UK and want someone to talk to:\n\n"
        f"{UK_RESOURCES}\n\n"
        "If you’d like, I can also share some coping ideas while you reach out."
    )

# ====== GGUF (llama.cpp) Chatbot ======

class Chatbot:
    """
    Fast CPU-friendly chatbot using a quantized TinyLlama GGUF model via llama-cpp-python,
    with TinyLlama's chat formatting and safety routing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_url: Optional[str] = None,
        n_ctx: int = int(os.getenv("N_CTX", "2048")),
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,  # keep 0 for CPU-only hosts
        system_prompt: Optional[str] = None,
    ):
        # Resolve model location (download if needed)
        self.model_path = self._ensure_model(model_path, model_url)
        self.system_prompt = system_prompt or THERAPY_SYSTEM
        self.history_text: Optional[str] = None

        if n_threads is None:
            n_threads = max(1, (os.cpu_count() or 2) - 1)

        from llama_cpp import Llama  # local import so the module can be imported without it

        print(f"[llama.cpp] Loading model: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"[llama.cpp] Ready (ctx={n_ctx}, threads={n_threads})")

    # ---- conversation helpers ----

    def reset_history(self):
        self.history_text = None

    def _prompt_with_history(self, user_text: str) -> str:
        if self.history_text is None:
            self.history_text = self.system_prompt
        return f"{self.history_text}<|user|>\n{user_text}\n<|end|>\n<|assistant|>\n"

    # ---- generation: non-streaming ----

    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "48")),
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        max_ctx_tokens: int = int(os.getenv("MAX_CTX_TOKENS", "384")),
    ) -> str:
        # Safety first
        if safety_check(user_text) == "crisis":
            return crisis_reply()
        if helpline_intent(user_text):
            return helpline_reply()

        prompt_text = self._prompt_with_history(user_text)

        # llama.cpp does context trimming internally if needed, but we'll keep prompts short anyway
        # Generate (completion API works fine for TinyLlama chat format)
        out = self.llm.create_completion(
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=["<|end|>"],
        )

        text = out["choices"][0]["text"].strip()
        # Update history (append assistant reply + end token)
        self.history_text = f"{prompt_text}{text}\n<|end|>\n"
        return text

    # ---- generation: streaming ----

    def stream_reply(
        self,
        user_text: str,
        max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "48")),
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
    ) -> Generator[str, None, None]:
        # Safety first (stream static messages immediately)
        if safety_check(user_text) == "crisis":
            yield crisis_reply()
            return
        if helpline_intent(user_text):
            yield helpline_reply()
            return

        prompt_text = self._prompt_with_history(user_text)

        stream = self.llm.create_completion(
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=["<|end|>"],
            stream=True,
        )

        partial = ""
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            partial += token
            if token:
                yield token
        # Save history
        self.history_text = f"{prompt_text}{partial}\n<|end|>\n"

    # ---- model download helper ----

    @staticmethod
    def _ensure_model(model_path: Optional[str], model_url: Optional[str]) -> str:
        """
        Ensure a GGUF model is present locally. If not provided, default to a good TinyLlama chat quant.

        Default URL: TinyLlama-1.1B-Chat-v1.0 Q4_K_M (by TheBloke)
        """
        # Prefer explicit path via env or arg
        model_path = (
            model_path
            or os.getenv("MODEL_PATH")
            or "/opt/models/tinyllama-1.1b-chat-q4_k_m.gguf"
        )
        model_file = pathlib.Path(model_path)

        if model_file.exists():
            return str(model_file)

        # Where to download from (override via env MODEL_URL)
        url = (
            model_url
            or os.getenv("MODEL_URL")
            or "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"
        )
        model_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] Fetching model to {model_file} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            size = int(r.headers.get("content-length", 0))
            bytes_read = 0
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_read += len(chunk)
                        if size:
                            pct = (bytes_read / size) * 100
                            if int(pct) % 10 == 0:
                                print(f"[download] {pct:.0f}%")
        print("[download] Done.")
        return str(model_file)
