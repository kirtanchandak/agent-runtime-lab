import requests

def chat(user_message):
    api_key = "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"
    model = "openrouter/free"
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "user", "content": user_message}]
    data = {
        "model": model,
        "messages": messages
    }
    response = requests.post(base_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Type a message and press Enter. Type 'quit' to exit.")
    while True:
        user_message = input("You: ").strip()
        if not user_message or user_message.lower() in {"quit", "exit"}:
            break
        print(chat(user_message))
        print()