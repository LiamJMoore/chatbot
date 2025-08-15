from chatbot import Chatbot

def main():
    bot = Chatbot()
    print("Bot initialised with:", bot.model_name)

    prompt = "Hello! How are you?"
    print("You:", prompt)

    # Show encoded prompt
    encoded = bot.encode_prompt(prompt)
    print("Encoded:", encoded)

    # Show decoded sample from token IDs
    decoded = bot.decode_reply([15496, 703, 345, 30])
    print("Decoded from token IDs:", decoded)

    # Generate a real reply
    print("Bot:", bot.generate(prompt))

if __name__ == "__main__":
    main()