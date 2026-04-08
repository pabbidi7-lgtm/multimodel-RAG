 taskset -c 0-7 python pipeline.py
2026-04-08 16:26:55.992218515 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2644117)
Waiting for pipeline to initialize...
Pipeline ready. Connecting client...

=== STEP 1: Basic text extraction ===
Starting ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.11s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.11s/doc]
Total time: 31.11 seconds

Results:  1
Failures: 0

=== STEP 1 SUCCEEDED ===
TestingDocument
A sample document with headings and placeholder text
Introduction
This is a placeholder document that can be used for any purpose. It contains some 
headings and some placeholder text to fill the space. The text is not important and contains 
no real value, but it is useful for testing. Below, we will have some simple tables and charts 
that we can use to confirm Ingest is working as expected.
Table 1
This table describes some animals, and some activities they might be do...

=== STEP 2: Full pipeline (extract + split + caption + embed + vdb) ===
Starting full ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:15<00:00, 15.09s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:15<00:00, 15.09s/doc]
WARNING:nv_ingest_client.client.interface:Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
Total time: 15.09 seconds

Results:  0
Failures: 1

=== STEP 2 FAILURES ===
--- [0] ---
('1:Docs/multimodal_test.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::34909df7-8b1f-4789-8583-32fcceab8ee6 -> Error in on_data: transform_text_split_and_tokenize_internal: error: TokenizersBackend has no attribute encode_plus')
Killed subprocess group 2644117
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ rm milvus db
rm: cannot remove 'milvus': No such file or directory
rm: cannot remove 'db': No such file or directory
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ rm milvus.db
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-04-08 16:30:15.294888700 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2652808)
Waiting for pipeline to initialize...
Pipeline ready. Connecting client...

=== STEP 1: Basic text extraction ===
Starting ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]
Total time: 31.12 seconds

Results:  1
Failures: 0

=== STEP 1 SUCCEEDED ===
Urine R/M Urine Sample
Accession No: DEMO_BARCODE Collected On: 21-Jan-25 13:40 Received On: 21-Jan-25 14:31 Approved On: 21-Jan-25 17:23
Observation Result Unit Biological Ref. Interval Method
Physical Examination
Urine Quantity 7.5 mL 7 - 8 Physical Examination
Urine Colour Pale Yellow Pale Yellow Physical Examination
Urinary Transparency Clear Clear Physical Examination
Biochemical Examination
Urinary pH 5.5 pH 6 .0 - 8.0 pH bromothymol blue
Urinary Specific Gravity 1.025 1.005 - 1.0...

=== STEP 2: Full pipeline (extract + split + caption + embed + vdb) ===
Starting full ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.17s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.17s/doc]
INFO:nv_ingest_client.util.vdb.milvus:46 elements to insert to milvus
INFO:nv_ingest_client.util.vdb.milvus:threshold for streaming is 1000
INFO:nv_ingest_client.util.vdb.milvus:streamed 46 records
INFO:nv_ingest_client.client.interface:Purging saved results from disk after successful VDB upload.
WARNING:nv_ingest_client.client.interface:Purge requested, but save_to_disk was not configured. No files to purge.
Total time: 65.87 seconds

Results:  1
Failures: 0

=== STEP 2 SUCCEEDED ===
Embeddings stored in Milvus Lite: milvus.db
Collection: medical_docs

=== STEP 3: Querying ingested documents ===
============================================================
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: What are all the test results that are outside the normal biological reference interval?
A: Based on the provided context, the following test results are outside the normal biological reference interval:

1. HbA1c: 5.7% (normal range: 4.8-5.7%, but according to the ADA guidelines, 5.7% is considered "at risk" or pre-diabetic)
2. 90 Day Average Blood Glucose: 116.89 mg/dl (normal range: 90-120 mg/dl, but this value is at the upper limit)
3. Blood Urea: 53 mg/dL (normal range: 19-43 mg/dL)
4. Blood Urea Nitrogen: 24.77 mg/dL (normal range: 9-20 mg/dL)
5. Creatinine: 1.41 mg/dL (normal range: 0.6-1.25 mg/dL)
6. Potassium: 5.2 mmol/L (normal range: 3.5-5.1 mmol/L, but this value is at the upper limit)
7. Triglyceride: 200 mg/dL (normal range: <150 mg/dL)
8. LDL Cholesterol: 102 mg/dL (normal range: <100 mg/dL)
9. Non-HDL Cholesterol: 142 mg/dL (normal range: <130 mg/dL)
10. Urinary Glucose: 1+ (normal range: Negative)
11. Red Cell Distribution Width (RDW-CV): 16.2% (normal range: 11.5-14.5%)
12. Red Cell Distribution Width (RDW-SD): 55.5 fL (normal range: 39-46 fL) 
13. A/G Ratio: 1.41 (normal range: 1.5-2.5) 
14. ESR: 9 mm/hr (normal range: <20 mm/hr, but this value is not necessarily abnormal, it's just at the higher end of the normal range)
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: Based on the kidney function test and eGFR classification table, what is the patient's GFR category?
A: Based on the kidney function test, the patient's estimated GFR (eGFR) is 57.10 mL/min/1.73m2. According to the eGFR classification table provided, this value falls into the category of G3a, which is classified as "Impaired Kidney Function".
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: What is the patient's HbA1c value and is this prediabetic or diabetic per ADA guidelines?
A: The patient's HbA1c value is 5.7%. According to the American Diabetes Association (ADA) guidelines provided in the context, an HbA1c value of 5.7-6.4% indicates that the patient is at risk (prediabetic).
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: Summarize the ultrasound whole abdomen findings and what tests were advised?
A: The ultrasound whole abdomen findings include:

1. Enlarged liver with increased parenchymal echogenicity, suggesting hepatomegaly with fatty changes (grade I-II).
2. Distended gallbladder with an intraluminal echogenic focus, possibly representing a polyp or sludge.
3. Normal pancreas, spleen, and kidneys, although the kidneys show increased cortical echogenicity and a lobulated outline.
4. A small umbilical hernia with herniation of omental fat.

The advised tests for correlation include:

1. Clinical correlation
2. Kidney Function Test (KFT)
3. Liver Function Test (LFT)
4. Elastography correlation

These tests are recommended to further investigate and confirm the findings from the ultrasound whole abdomen and to rule out any potential complications or underlying conditions.
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: What are the lipid profile results and classify each as optimal, borderline high, or high?
A: Based on the provided context, the lipid profile results are as follows:

1. Total Cholesterol: 192 mg/dL
   - Classification: Borderline High (according to the clinical decision limits, optimal is <200, and this value is close to but still below 200)

2. Triglyceride: 200 mg/dL
   - Classification: Borderline High (according to the clinical decision limits, optimal is <150, borderline high is 150-199, and this value is exactly at the start of the borderline high range but given the exact value, it's more accurately described as the threshold for borderline high)

3. HDL Cholesterol: 50 mg/dL
   - Classification: Optimal (according to the clinical decision limits, optimal is >45)

4. VLDL Cholesterol: 40 mg/dL
   - Classification: Optimal (according to the biological reference interval, the range is 5-40, and this value is within the optimal range)

5. LDL Cholesterol: 102 mg/dL
   - Classification: Borderline High (according to the clinical decision limits, optimal is <100, and this value is slightly above the optimal range)

6. Non-HDL Cholesterol: 142 mg/dL
   - Classification: Borderline High (according to the clinical decision limits, optimal is <130, and this value is above the optimal range)

7. LDL/HDL Ratio: 2.04
   - Classification: Optimal (according to the calculated ratio, the range is 1.5-3.5, and this value is within the optimal range)

8. TC/HDL Ratio: 3.84
   - Classification: Optimal (according to the calculated ratio, the range is 3-5, and this value is within the optimal range)

Therefore, the classifications are:
- Total Cholesterol: Borderline High
- Triglyceride: Borderline High
- HDL Cholesterol: Optimal
- VLDL Cholesterol: Optimal
- LDL Cholesterol: Borderline High
- Non-HDL Cholesterol: Borderline High
- LDL/HDL Ratio: Optimal
- TC/HDL Ratio: Optimal
------------------------------------------------------------
Killed subprocess group 2652808
E20260408 16:37:08.189358 2660435 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-04-08 16:41:10.289542209 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2670986)
Waiting for pipeline to initialize...
Pipeline ready. Connecting client...

=== STEP 1: Basic text extraction ===
Starting ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]
Total time: 31.12 seconds

Results:  1
Failures: 0

=== STEP 1 SUCCEEDED ===


=== STEP 2: Full pipeline (extract + split + caption + embed + vdb) ===
Starting full ingestion...
Processing:   0%|                                                                                                | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.17s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.17s/doc]
INFO:nv_ingest_client.util.vdb.milvus:46 elements to insert to milvus
INFO:nv_ingest_client.util.vdb.milvus:threshold for streaming is 1000
INFO:nv_ingest_client.util.vdb.milvus:streamed 46 records
INFO:nv_ingest_client.client.interface:Purging saved results from disk after successful VDB upload.
WARNING:nv_ingest_client.client.interface:Purge requested, but save_to_disk was not configured. No files to purge.
Total time: 66.82 seconds

Results:  1
Failures: 0

=== STEP 2 SUCCEEDED ===
Embeddings stored in Milvus Lite: milvus.db
Collection: multimodal_docs

=== STEP 3: Querying ingested documents ===
============================================================
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: What is the name mentioned in the ID
A: The patient ID mentioned is UHID.DEMO.001. The name mentioned in the ID is "Demo".
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: what is the service number mentioned?
A: The context does not mention a "service number". It does mention a "Lab No" and a "Visit No", but the values for these are listed as "Demo", which suggests that they are placeholders or sample values rather than actual numbers. Additionally, it mentions a "Patient ID" which is "UHID.DEMO.001".
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: What is the place birth?
A: The place of birth is not mentioned in the context.
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: what is her data of birth?
A: The answer is not in the context. The provided context does not mention the patient's date of birth. It only mentions the patient's age (60 years old) and the dates of various medical tests and procedures.
------------------------------------------------------------
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Q: which country is she belongs to?
A: The answer is not in the context. The context does not provide information about the patient's country of origin. It only provides details about the patient's medical report, test results, and demographic information such as age and sex, but not nationality or country of origin.
------------------------------------------------------------
Killed subprocess group 2670986
E20260408 16:47:11.949846 2678817 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
