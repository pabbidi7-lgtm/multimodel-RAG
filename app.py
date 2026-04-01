def rerank_passages(query, passages):
    if not passages:
        return []

    data = _api_call(
        RERANK_URL,
        {
            "model": RERANK_MODEL,
            "input": [
                {
                    "query": query,
                    "passages": passages
                }
            ],
        },
    )

    # NVIDIA returns nested format
    try:
        return data["data"][0]["rankings"]
    except Exception:
        return []
