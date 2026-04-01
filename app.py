taskset -c 0-7 python rag_agent.py --demo --demo-pdf ./Docs/minion-tech.pdf

+===========================================================+
|  NV-Ingest 25.9.0 + LangGraph RAG Agent                  |
|                                                           |
|  Nodes: classifier -> retriever -> reranker -> generator  |
|  Retry: LOW confidence x1  |  hallucination x1           |
+===========================================================+

  OK NVIDIA_API_KEY: nvapi-gCbDQq9wb...
  OK Milvus DB: ./milvus_rag.db
  OK Collection: rag_documents
  OK Reranker URL: https://integrate.api.nvidia.com/v1/ranking

============================================================
  DEMO MODE
============================================================
  > Document: minion-tech.pdf


============================================================
  STARTING NV-INGEST PIPELINE
============================================================
  > Importing NV-Ingest (loads Ray internally)...
2026-04-01 15:08:21.948661891 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (7.6s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (15.7s)
  > Ingesting 1 file(s)...
  >   -> minion-tech.pdf (10791 KB)
  ERR Demo failed: (Schema Error): Extra inputs are not permitted
 -> {"tokenizer": "meta-llama/Llama-3.2-1B", "chunk_size": 512, "chunk_overlap": 50, split_by: "token", "params": {"split_source_types": ["text", "table", "chart"], "hf_access_token": ""}}
Demo failed
Traceback (most recent call last):
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/processing.py", line 231, in check_schema
    return schema(**options)
           ^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pydantic/main.py", line 250, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for SplitTaskSchema
split_by
  Extra inputs are not permitted [type=extra_forbidden, input_value='token', input_type=str]
    For further information visit https://errors.pydantic.dev/2.12/v/extra_forbidden

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/rag_agent.py", line 839, in main
    run_demo(pdf)
  File "/home/clouduser01/jaswanth/rag_agent.py", line 703, in run_demo
    info = run_ingest([pdf_path])
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/jaswanth/rag_agent.py", line 360, in run_ingest
    .split(
     ^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 87, in wrapper
    return func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 813, in split
    task_options = check_schema(SplitTaskSchema, kwargs, "split", json.dumps(kwargs))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/processing.py", line 234, in check_schema
    raise ValueError(error_message) from e
ValueError: (Schema Error): Extra inputs are not permitted
 -> {"tokenizer": "meta-llama/Llama-3.2-1B", "chunk_size": 512, "chunk_overlap": 50, split_by: "token", "params": {"split_source_types": ["text", "table", "chart"], "hf_access_token": ""}}
Killed subprocess group 329039
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
