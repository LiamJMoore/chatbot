from typing import Optional
import importlib

try:
    torch = importlib.import_module("torch")
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


class Chatbot:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
        device: Optional[str] = None,
        tokenizer=None,
        model=None,
        autoload: bool = True,
    ):
        self.model_name = model_name
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.tokenizer = None
            self.model = None

        if device is None:
            device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self.device = device

        if self.model is not None and _HAS_TORCH:
            self.model.to(self.device)

    def encode_prompt(self, prompt: str):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Provide one or set autoload=True.")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for return_tensors='pt'. Install with: pip install torch")
        return self.tokenizer(prompt, return_tensors="pt")

    def decode_reply(self, reply_ids: list[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Provide one or set autoload=True.")
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

    def generate_reply(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded. Provide them or set autoload=True.")
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to generate. Install with: pip install torch")

        prompt = prompt + "\n"
        encoded = self.encode_prompt(prompt)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=50
        )

        decoded_text = self.decode_reply(output_ids[0])
        return decoded_text[len(prompt):].strip()
