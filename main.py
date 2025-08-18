import os
import re
import shutil
import textwrap
from chatbot import Chatbot, UK_RESOURCES

DISCLAIMER = (
    "I’m not a therapist or crisis service. I can listen, share information, "
    "and point you to resources. For personalised care, speak to a qualified professional. "
    "If you’re in immediate danger, call 999 now."
)

# Put guidance in system_prompt (NOT before each user message)
MENTAL_HEALTH_SYSTEM = (
    "<|system|>\n"
    "You are a supportive, evidence-informed mental health guide.\n"
    "DO: answer clearly; explain concepts (anxiety, depression, PTSD), common symptoms, "
    "risk/protective factors; give practical low-risk coping skills (CBT/DBT/mindfulness/"
    "behavioural activation); suggest how to seek help (GP/therapist, NHS Talking Therapies/IAPT, helplines).\n"
    "DO: use plain English, short paragraphs, and optional bullet points.\n"
    "DON'T: diagnose, prescribe, create treatment plans, or make life/death judgments.\n"
    "IMPORTANT: Do NOT include crisis hotlines or crisis language unless the user's message itself indicates "
    "self-harm, suicide, or immediate danger. For non-crisis (e.g., greetings, 'anxiety attack'), provide skills first.\n"
    "End with one gentle next step.\n"
    "<|end|>\n"
)

# Strict crisis trigger: explicit phrases only
CRISIS_PAT = re.compile(
    r"(?:^|\b)(?:kill myself|suicide|self[- ]?harm|hurt myself|end (?:it|my life)|"
    r"take my life|overdose|i don['’]t want to be alive|dont want to be alive|"
    r"not be in this world|life (?:is|feels) not worth)(?:\b|$)",
    re.IGNORECASE,
)

# Non-crisis panic first-aid trigger
PANIC_PAT = re.compile(
    r"\b(panic attack|anxiety attack|having (?:a )?panic|can[’']?t breathe|hyperventilat(?:e|ing))\b",
    re.IGNORECASE,
)

CRISIS_TEMPLATE = (
    "I’m really glad you told me. I’m not a therapist, but I care about your safety.\n\n"
    "Are you in immediate danger right now? If yes, please call 999 or go to the nearest A&E.\n\n"
    "If you can, tell someone you trust nearby — you don’t have to do this alone.\n\n"
    "UK 24/7 support (free & confidential):\n"
    "• Samaritans — 116 123 or jo@samaritans.org\n"
    "• Shout — text 85258\n"
    "• NHS 111 — urgent medical help when it’s not life-threatening\n\n"
    "I can stay with you here. Would grounding tips for the next few minutes help, "
    "or would you like help deciding who to contact first?"
)

PANIC_FIRST_AID = (
    "I’m here with you. Panic peaks and passes. Let’s do a quick 2-minute reset:\n\n"
    "Breathing (box breath · 4-4-4-4):\n"
    "• Inhale through your nose for 4\n"
    "• Hold for 4\n"
    "• Exhale slowly for 4\n"
    "• Hold for 4 — repeat 6–8 cycles\n\n"
    "Grounding (5-4-3-2-1):\n"
    "• 5 things you can see\n"
    "• 4 things you can feel (clothes, chair, floor)\n"
    "• 3 things you can hear\n"
    "• 2 things you can smell\n"
    "• 1 thing you can taste\n\n"
    "Muscle reset (shoulders/jaw/hands): tense for 5, release for 10 — repeat twice.\n\n"
    "If you want, tell me what triggered this one — we can plan a small step for next time."
)

def term_width(default=80):
    return shutil.get_terminal_size((default, 20)).columns

def wrap(text: str, width: int | None = None) -> str:
    w = width or term_width()
    return "\n\n".join(textwrap.fill(p, width=w) for p in text.split("\n\n"))

def is_crisis(msg: str) -> bool:
    return bool(CRISIS_PAT.search(msg or ""))

def is_panic(msg: str) -> bool:
    return bool(PANIC_PAT.search(msg or ""))

def main():
    model_path = os.getenv("MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # IMPORTANT: set system_prompt here so Chatbot uses it in history
    bot = Chatbot(model_path=model_path, system_prompt=MENTAL_HEALTH_SYSTEM)

    print("=== Mental Health Info Bot (not a crisis service) ===")
    print(wrap(DISCLAIMER))
    print("\nGet help now:\n" + wrap(UK_RESOURCES) + "\n")
    print("Commands: 'reset' (clear memory), 'help' (resources), 'exit' (quit).\n")

    while True:
        try:
            user = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTake care. Goodbye.")
            break

        if not user:
            continue

        low = user.lower()
        if low in {"exit", "quit"}:
            print("Take care. Goodbye.")
            break
        if low == "help":
            print("\n" + wrap(UK_RESOURCES) + "\n")
            continue
        if low == "reset":
            bot.reset_history()
            print("Bot: (memory cleared)")
            continue

        # 1) Crisis route (only if user's OWN text matches)
        if is_crisis(user):
            print("Bot:")
            print(wrap(CRISIS_TEMPLATE))
            print()
            continue

        # 2) Panic/anxiety first-aid
        if is_panic(user):
            print("Bot:")
            for paragraph in PANIC_FIRST_AID.split("\n\n"):
                print(wrap(paragraph))
                print()
            continue

        # 3) Normal MH Q&A path — pass ONLY the raw user text
        try:
            reply = bot.generate_reply(user)  # <-- no prefixed guidance here
        except Exception as e:
            print(f"Bot error: {e}")
            continue

        print("Bot:")
        print(wrap(reply))
        print()

if __name__ == "__main__":
    main()
