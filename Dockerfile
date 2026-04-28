Yes — I can tell what is happening from your logs.
Short answer
This is not mainly a Milvus corruption issue and not mainly because the documents failed to ingest.
The main problem is:

The documents are ingesting successfully
Retrieval is picking chunks from the wrong document
All files are going into one shared collection
You are not filtering retrieval by source document
Your model is then answering based on the wrong context
In one case, the model also adds outside knowledge / mild hallucination


What the logs already prove
1) The documents are ingesting
Your logs clearly show these files were successfully processed and uploaded:

Docs/Oxford.pdf → 435 elements inserted
Docs/merger_agreement 1.pdf → 23 elements inserted
Docs/policy-2.pdf → 23 elements inserted

So those PDFs are ingested. The issue is not “document not ingested” for those three.
Also the batch summary shows:

Total files: 15
OK: 10
Skipped: 5
Failed: 0

That means the merger PDF and Oxford PDF were not skipped, and they were not failed. They were successfully embedded and uploaded.

2) Why is it showing “medical document” for a merger/Oxford question?
Because retrieval is returning chunks from policy-2.pdf instead of from:

merger_agreement 1.pdf
Oxford.pdf

You already showed earlier metrics like this:

top_sources = policy-2.pdf for the merger question
top_sources = policy-2.pdf for the WCAG question

That is the biggest clue.
So the chain is:
What is happening

You ask a question about merger_agreement 1.pdf
Milvus retrieval returns chunks from policy-2.pdf
The LLM sees only those policy chunks
So it says:
“the provided context appears to be related to medical documentation…”

That is actually the model honestly describing the wrong retrieved context.
So the phrase “medical document” is not random.
It means the retriever gave the LLM the wrong document chunks.

Direct answers to your questions
“Is this hallucination?”
For the merger answer
Mostly no hallucination.
It is mainly a wrong-context retrieval problem.
The model is saying:

“I see medical policy context”
“I do not see merger content”

That is actually consistent with the context it got.
For the WCAG answer
Yes, partial hallucination / fallback knowledge is happening.
Because it says:

answer not in context
then gives general WCAG knowledge anyway

That means your prompt is not strict enough to force a clean “not found” only.
So:

Merger answer: mostly retrieval failure
WCAG answer: retrieval failure + model using outside knowledge


“Are the documents not ingesting?”
No — for merger/Oxford/policy, they are ingesting
The logs prove ingestion succeeded.
But image-only files are not ingesting into useful text
These were explicitly skipped:

Driving.jpg
Screenshot (1).png
Singapore_NID_B 1.jpeg
Singapore_NID_F 1.jpeg
california-drivers-license-small 1 1.jpg

So any question that depends on those image files will fail in the current setup unless OCR NIM is running.
But this is not the reason your merger/Oxford questions failed, because those PDFs were ingested.

“Any question related to non-ingested document?”
For the 3 example questions you tested now:
Merger question

related to merger_agreement 1.pdf
that file was ingested

WCAG question

related to Oxford.pdf
that file was ingested

Discharge summary question

related to policy-2.pdf
that file was ingested

So these questions are not about non-ingested documents.

“Before I was ingesting one PDF and asking 5 questions — it worked. Now 15 at a time, why broken?”
This is the most important point.
When you ingested only one PDF
Milvus had only one document’s chunks to choose from.
So:

retrieval had no confusion
every query was forced to match that one document
answers looked correct

Now you ingest 15 files into one collection
Now retrieval has to choose from all document chunks together.
That creates these problems:
A. Cross-document contamination
A question about one PDF may retrieve another PDF.
That is exactly what happened:

merger question → policy chunks
WCAG question → policy chunks

B. No source filtering
You are querying one single collection:
PythonCOLLECTION_NAME = "multimodal_docs"Show more lines
and retrieving globally.
So Milvus is not being told:

“search only in merger_agreement 1.pdf”
“search only in Oxford.pdf”

Without source filtering, retrieval can return any chunk from any file.
C. Dense retrieval may semantically confuse documents
Because you are doing dense retrieval, queries can sometimes match generic language from the wrong document.
For example:

policy text is structured and explicit
merger legal text is denser and more specific
Oxford accessibility text may be split across pages and figure explanations

Dense search can accidentally rank “wrong but semantically decent” policy chunks above the exact merger/Oxford chunk.
D. Long multi-part questions are harder
Your question style is moderate-hard and multi-part, e.g.:

“how are X, Y, Z treated differently, and what does PDI receive…”

That requires retrieving a chunk containing multiple related facts.
If chunking or ranking misses that section, the wrong source wins.

Is Milvus DB faulty?
Probably not faulty in the sense of “broken database”
Milvus is doing what you asked:

storing all chunk embeddings
retrieving nearest neighbors

The problem is how the data is organized and queried, not that Milvus is damaged.
But there is one important Milvus-related issue:
You are probably reusing the same DB / same collection across runs
Your logs show the same:

milvus.db
multimodal_docs

If you are not clearing the collection before each fresh experiment, then:

old chunks remain
duplicate chunks remain
previous runs accumulate
retrieval quality can degrade

So not “Milvus fault,” but collection contamination / stale data accumulation is very possible.

Exact diagnosis from your case
Why policy questions work
Because the retriever is strongly hitting policy-2.pdf, and the policy questions are indeed about that file.
So:

correct file retrieved
answer comes out correct

Why merger/Oxford fail
Because the retriever is incorrectly returning policy-2.pdf chunks for those questions.
So:

wrong file retrieved
answer says “medical policy” or “not in context”
or gives outside knowledge

That is the root cause.

So what is the actual problem?
Main problem summary
Your current multi-file setup has retrieval routing failure.
Not mainly:

not missing ingestion of merger/Oxford
not pure hallucination
not Milvus corruption

Mainly:

wrong-document retrieval
all docs mixed in one collection
no metadata/source filtering
likely stale collection reuse
prompt not strict enough to block outside knowledge


Why did the discharge summary answer come correct?
Because the retriever pulled the correct document (policy-2.pdf) for that question.
So your pipeline is not completely broken.
It is working when retrieval lands on the right source.
That tells us:

ingestion is okay
embeddings are being stored
Milvus is retrievable
LLM answering works

The weak part is document targeting in retrieval.

Also note this important thing from your run
You used:
Shelltaskset -c 0-7 python pipeline.pyShow more lines
That only restricts CPU affinity.
It does not solve retrieval contamination.
It only helps the pipeline run stably.
So taskset fixed the execution/freezing issue, but not the multi-document retrieval quality problem.
