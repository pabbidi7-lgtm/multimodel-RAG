 python app.py
2026-03-26 04:57:10.037221140 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 32.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Applying hyperthread weight (0.75) to logical limit 32 (System: 16P/32L): Effective weighted cores = 28.00
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 28.00 (Method: sched_affinity_weighted)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Starting pipeline setup
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_pipeline:PIDController initialized using ScalingConfig.
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 32.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Applying hyperthread weight (0.75) to logical limit 32 (System: 16P/32L): Effective weighted cores = 28.00
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 28.00 (Method: sched_affinity_weighted)
INFO:nv_ingest_api.util.system.hardware_info:Applying hyperthread weight (0.75) to logical limit 32 (System: 16P/32L): Effective weighted cores = 28.00
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pid_controller:[ConstraintMgr] Initialized. MaxReplicas=1, EffectiveCoreLimit=28.00 (Method: sched_affinity_weighted), CoreBasedReplicaLimit=28, MemThreshold=96607MB, EffectiveLimit=96607.0MB 
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_pipeline:ResourceConstraintManager initialized using ScalingConfig.
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_stat_collector:RayStatsCollector initialized (Interval: 10.0s, Actor Timeout: 5.0s, Queue Timeout: 2.0s, EMA Alpha: 0.1)
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_pipeline:RayStatsCollector initialized using StatsConfig.
Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/app.py", line 21, in <module>
    result = run_pipeline(pipeline_config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest/framework/orchestration/ray/util/pipeline/pipeline_runners.py", line 382, in run_pipeline
    pipeline, total_elapsed = _launch_pipeline(
                              ^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest/framework/orchestration/ray/util/pipeline/pipeline_runners.py", line 256, in _launch_pipeline
    _ = setup_ingestion_pipeline(pipeline, ingest_config.model_dump())
                                           ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'model_dump'
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_pipeline:Stopping pipeline...
INFO:nv_ingest.framework.orchestration.ray.primitives.ray_pipeline:Pipeline stopped successfully.
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
