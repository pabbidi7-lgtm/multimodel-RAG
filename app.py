sed -i '/from dotenv import load_dotenv/d' pipeline.py
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ sed -i '/load_dotenv()/d' pipeline.py
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ export NVIDIA_API_KEY=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
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ export NVIDIA_BUILD_API_KEY=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
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-03-26 18:04:16.446293658 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1774548258.368580 2299067 dns_resolver_ares.cc:358] no server name supplied in dns URI
E0000 00:00:1774548258.368671 2299067 legacy_channel.cc:89] channel stack builder failed: UNKNOWN: the target uri is not valid: dns:///
^CTraceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline.py", line 70, in <module>
    milvus = MilvusClient(uri=MILVUS_URI)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/milvus_client/milvus_client.py", line 88, in __init__
    self._handler = self._manager.get_or_create(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/connection_manager.py", line 472, in get_or_create
    return self._create_shared(config, client, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/connection_manager.py", line 496, in _create_shared
    handler._wait_for_channel_ready(timeout=timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/grpc_handler.py", line 225, in _wait_for_channel_ready
    grpc.channel_ready_future(self._channel).result(timeout=timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/grpc/_utilities.py", line 162, in result
    self._block(timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/grpc/_utilities.py", line 102, in _block
    self._condition.wait()
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/threading.py", line 355, in wait
    waiter.acquire()
