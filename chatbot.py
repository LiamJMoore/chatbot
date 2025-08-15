from typing import Optional
import importlib
import re

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
_CRISIS_RE = [re.compile(pat, re.IGNORECASE) for pat in CRISIS_PATTERNS]

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

# ===== Lazy import torch so this module imports without it =====

try:
    torch = importlib.import_module("torch")
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

def _select_device() -> str:
    if not _HAS_TORCH:
        return "cpu"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except Exception:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


class Chatbot:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        tokenizer=None,
        model=None,
        autoload: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or THERAPY_SYSTEM
        self.history_text: Optional[str] = None

        if device is not None and _HAS_TORCH:
            self.device = torch.device(device)
        else:
            self.device = _select_device()

        if _HAS_TORCH:
            dtype_label = getattr(self.device, "type", "cpu")
            print(f"[Device] Using {'GPU' if dtype_label in ('cuda','mps') else 'CPU'}: {self.device}")
        else:
            print("[Device] PyTorch not installed; using CPU.")

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        elif autoload:
            try:
                transformers = importlib.import_module("transformers")
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "transformers is required when autoload=True. Install with: pip install transformers"
                ) from e

            AutoTokenizer = getattr(transformers, "AutoTokenizer")
            AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # ---- IMPORTANT: avoid meta tensors on CPU; materialize real weights ----
            model_kwargs = {}
            if _HAS_TORCH:
                dev_type = getattr(self.device, "type", str(self.device))
                if dev_type == "cuda":
                    model_kwargs["torch_dtype"] = getattr(torch, "float16", None)
                    model_kwargs["low_cpu_mem_usage"] = False
                    model_kwargs["device_map"] = None
                else:  # CPU (or MPS -> load on CPU then .to(mps) if needed)
                    model_kwargs["torch_dtype"] = torch.float32
                    model_kwargs["low_cpu_mem_usage"] = False
                    model_kwargs["device_map"] = None

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if _HAS_TORCH:
                self.model = self.model.to(self.device)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = True
            if hasattr(self.model, "eval"):
                self.model.eval()
        else:
            self.tokenizer = None
            self.model = None

        # Ensure pad token exists (often EOS)
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache TinyLlama's <|end|> id if present
        self.end_token_id = None
        if self.tokenizer is not None:
            end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
            if isinstance(end_id, int) and end_id != self.tokenizer.unk_token_id:
                self.end_token_id = end_id

    # --- utilities ---
    def reset_history(self):
        self.history_text = None

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.reset_history()

    def encode_prompt(self, text: str):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for return_tensors='pt'. Install with: pip install torch")
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode_reply(self, reply_ids: list[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

    def _prompt_with_history(self, user_text: str) -> str:
        if self.history_text is None:
            self.history_text = self.system_prompt
        return f"{self.history_text}<|user|>\n{user_text}\n<|end|>\n<|assistant|>\n"

    # --- chat generation with safety + token-accurate slicing ---
    def generate_reply(
        self,
        user_text: str,
        max_new_tokens: int = 80,   # lighter default for small plans
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> str:
        # 1) Safety short-circuits
        if safety_check(user_text) == "crisis":
            return crisis_reply()
        if helpline_intent(user_text):
            return helpline_reply()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded.")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to generate. Install with: pip install torch")

        prompt_text = self._prompt_with_history(user_text)
        enc = self.encode_prompt(prompt_text)

        eos_id = self.end_token_id or self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # Slice by token IDs (not characters)
        start = enc["input_ids"].shape[-1]
        gen_ids = out[0, start:]
        reply_text = self.decode_reply(gen_ids.tolist()).strip()

        # Update history for next turn
        self.history_text = f"{prompt_text}{reply_text}\n<|end|>\n"
        return reply_text

    # Optional one-shot generator (non-chat)
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded.")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to generate. Install with: pip install torch")
        enc = self.encode_prompt(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.end_token_id or self.tokenizer.eos_token_id,
            )
        ids = out[0, enc["input_ids"].shape[-1]:].tolist()
        return self.decode_reply(ids)
