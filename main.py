from chatbot import Chatbot, UK_RESOURCES

DISCLAIMER = (
    "I’m not a therapist or crisis service. I can listen, reflect, and share resources.\n"
    "If you’re in immediate danger or feel unable to stay safe, call 999 now."
)

def main():
    bot = Chatbot(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("=== Support Bot (not a crisis service) ===")
    print(DISCLAIMER)
    print("\nGet help now:\n" + UK_RESOURCES + "\n")
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
            print("\n" + UK_RESOURCES + "\n")
            continue
        if low == "reset":
            bot.reset_history()
            print("Bot: (memory cleared)")
            continue

        try:
            reply = bot.generate_reply(user)
        except Exception as e:
            print(f"Bot error: {e}")
            break

        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    main()
