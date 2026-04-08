llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

print("\n" + "=" * 60)
print("RAG Chatbot Ready! Ask your questions (type 'exit' to quit)")
print("=" * 60)

while True:
    try:
        q = input("\nAsk a question: ").strip()

        if q.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not q:
            continue

        retrieved_docs = nvingest_retrieval(
            [q],
            collection_name,
            milvus_uri=milvus_uri,
            hybrid=sparse,
            top_k=10,
        )

        if retrieved_docs and retrieved_docs[0]:
            context = "\n\n".join([doc["entity"]["text"] for doc in retrieved_docs[0]])
        else:
            context = "No relevant content found."

        prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {q}
Answer:"""

        completion = llm_client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )

        print(f"\nA: {completion.choices[0].message.content}")

    except KeyboardInterrupt:
        print("\nExiting...")
        break
