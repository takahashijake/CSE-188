import ollama

MODEL = "qwen2.5:3b"  # change this when you switch to GPU models

print(f"Chatting with {MODEL}. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": user_input}],
        options={"temperature": 0}
    )

    print(f"\nModel: {response['message']['content']}\n")