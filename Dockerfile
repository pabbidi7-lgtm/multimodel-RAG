(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ ray stop --force
Did not find any active Ray processes.
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ pkill -f "ray::"
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ pkill -f "nv_ingest"
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ sleep 3
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ rm milvus_rag.db
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ lscpu | grep -E "CPU\(s\)|Thread|Core|Socket"
CPU(s):                                  32
On-line CPU(s) list:                     0-31
Thread(s) per core:                      2
Core(s) per socket:                      16
Socket(s):                               1
NUMA node0 CPU(s):                       0-31
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-3 python pipeline2.py
2026-04-08 15:13:13,929 INFO worker.py:2004 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/ray/_private/worker.py:2052: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
2026-04-08 15:13:21.104327330 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 4 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 4.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 4.00 (Method: sched_affinity)
INFO:__main__:Created collection
INFO:__main__:Ingesting: ['./Docs/multimodal_test.pdf']
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2552661)
INFO:__main__:Pipeline started...
INFO:__main__:Waiting for broker localhost:7671…
INFO:__main__:Broker ready!
Processing:   0%|                                                                          | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
