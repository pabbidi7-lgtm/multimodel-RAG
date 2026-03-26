NVIDIA_API_KEY=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
NVIDIA_BUILD_API_KEY=nvbuild-gCbDQq9wbFj-DF7-CfmpCODd4VNX2JsB0Zfvj0r9HyMw_5YGOIOFdmA-oHGY-rZ3







 export NVIDIA_API_KEY=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
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ export NVIDIA_BUILD_API_KEY=nvbuild-gCbDQq9wbFj-DF7-CfmpCODd4VNX2JsB0Zfvj0r9HyMw_5YGOIOFdmA-oHGY-rZ3
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-03-26 17:51:07.589634875 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1774547469.419145 2293945 dns_resolver_ares.cc:358] no server name supplied in dns URI
E0000 00:00:1774547469.419236 2293945 legacy_channel.cc:89] channel stack builder failed: UNKNOWN: the target uri is not valid: dns:///
