curl -X POST https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": ["test"], "model": "nvidia/nv-embedqa-e5-v5", "input_type": "query"}' \
  | python3 -m json.tool
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   240  100   162  100    78    504    243 --:--:-- --:--:-- --:--:--   750
{
    "status": 404,
    "title": "Not Found",
    "detail": "Function '09c64e32-2b65-4892-a285-2f585408d118': Not found for account '38FN-XDcco9msc2defg-jUydEzvtOLO0B4O0UlrcXzw'"
}






export NVIDIA_API_KEY=nvapi-<new-key-from-build.nvidia.com>
curl -X POST https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": ["test"], "model": "nvidia/nv-embedqa-e5-v5", "input_type": "query"}' \
  | python3 -m json.tool


curl -X POST https://ai.api.nvidia.com/v1/cv/nvidia/table-structure-recognition \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"test": "ping"}' \
  | python3 -m json.tool
