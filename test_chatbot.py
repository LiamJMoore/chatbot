from chatbot import Chatbot

class FakeTokenizer:
    eos_token = ""
    eos_token_id = 0

def test_init_with_injected_dependencies():
    fake_tok = FakeTokenizer()
    fake_model = object()
    bot = Chatbot(tokenizer=fake_tok, model=fake_model, autoload=False)
    assert bot.model_name == "microsoft/DialoGPT-small"
    assert bot.tokenizer is fake_tok
    assert bot.model is fake_model
