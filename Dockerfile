taskset -c 0-3 python pipeline2.py
2026-04-08 15:38:24.057659628 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 4 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 4.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 4.00 (Method: sched_affinity)
INFO:__main__:Collection exists
INFO:__main__:Ingesting: ['./Docs/multimodal_test.pdf']
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775662705.861694 2566937 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2567165)
INFO:__main__:Pipeline started...
INFO:__main__:Waiting for broker localhost:7671...
Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline2.py", line 183, in <module>
    chunks = ingest_document([pdf], output_dir="./temp_ingest")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/jaswanth/pipeline2.py", line 70, in ingest_document
    wait_for_broker()
  File "/home/clouduser01/jaswanth/pipeline2.py", line 47, in wait_for_broker
    raise RuntimeError("Broker timeout")
RuntimeError: Broker timeout
Killed subprocess group 2567165
E20260408 15:40:26.775457 2567134 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
