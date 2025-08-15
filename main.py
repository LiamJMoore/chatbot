from chatbot import Chatbot

if __name__ == "__main__":
    bot = Chatbot()

    prompts = [
        "What's your name?",
        "What do you think about AI?",
        "Sorry, tell me your name again."
    ]

    for p in prompts:
        reply = bot.generate_reply(p)
        print(f"Prompt: {p}")
        print(f"Reply: {reply}\n")

    # Example of resetting history
    # bot.reset_history()
    # print("History reset.")
    # print("Reply after reset:", bot.generate_reply("Who are you?"))
