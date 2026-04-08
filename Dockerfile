print("\n--- RAG Chatbot Ready ---")
print("Type your question (or 'exit' to quit)\n")

while True:
    try:
        q = input("Ask a question: ").strip()

        if q.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not q:
            continue

        answer = rag_chatbot(q)
        print(f"\nA: {answer}\n")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
