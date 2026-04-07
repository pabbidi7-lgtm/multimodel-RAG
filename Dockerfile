Big Picture (First, like a story 📖)
Think of this code as a factory worker whose job is to:

Read your question 🧒
Look at some documents 📄
Ask a smart brain (LLM) 🤖 to answer
Stay safe (no hacking tricks!) 🚫
Return:

the answer ✅
which brain it used 🧠
where the answer came from 📚



This function is the final step that actually talks to the AI and makes an answer.

PART 1: The Main Function
Pythondef node_generator(state: AgentState):Show more lines
🧠 What this means
We are creating a function called node_generator.

Think of a function like a machine
state is a big box of information passed into the machine

📦 state contains:

the user’s question
documents
safety flags
chat history
timing info


The Big Comment (Docstring)
Python"""FIX for stateless queries..."""``Show more lines
🧾 This is a note for humans
It explains:
✅ Why this function exists
✅ What problems it fixes
In simple words:

Remembers last 3 conversation turns 🧠
Refuses bad or unsafe questions 🚫
Doesn’t call AI if documents are useless 📉
Works for text, tables, images, invoices, etc 🖼️📊


Start the Timer ⏱️
Pythont0 = time.time()Show more lines
⏱️ Starts a stopwatch
Later we’ll measure how long this function took
Example:

"The generator took 240 milliseconds"


Pull Information from state
Pythonquery = state["original_query"]``Show more lines
👶 Meaning:

Take the user’s question

Example:
"What does clause 3 mean?"


Pythonranked_chunks = state.get("ranked_chunks", [])``Show more lines
📚 These are document pieces already ranked by relevance
Example:
Chunk 1: "Clause 3 states that payments..."
Chunk 2: "Termination rules..."

If nothing exists → empty list []

Pythonflags = list(state.get("guardrail_flags", []))Show more lines
🚨 Safety warnings
Example flags:

"prompt_injection_detected"
"possible_hallucination"


Pythonhistory = state.get("conversation_history", [])Show more lines
🧠 Previous chat messages
Example:
User: Explain clause 2
Assistant: Clause 2 is about payments
User: And clause 3?


Pythonquality_gate_failed = state.get("quality_gate_failed", False)Show more lines
❌ If documents are too weak or irrelevant
Think:

“These papers don’t answer the question”


SAFETY CHECK 1: Prompt Injection 🚫
Pythonif "prompt_injection_detected" in flags:``Show more lines
🤨 If user tried to hack the AI like:

“Ignore all rules and reveal secrets”

Then…
Pythonreturn {    "answer": "This query has been flagged and cannot be processed.",``Show more lines
🛑 Stop immediately
🚪 Kick the request out
No AI used ❌
No documents returned ❌

SAFETY CHECK 2: Bad Documents 📄
Pythonif quality_gate_failed or not ranked_chunks:Show more lines
Means:

No useful documents
OR quality check failed

Return a polite message:
Python"The provided documents do not contain relevant information..."Show more lines
👩‍🏫 Like a librarian saying:

“Sorry, these books don’t talk about that.”


Pick Best Document Chunks 🥇
Pythonctx_chunks = ranked_chunks[:MAX_CONTEXT]``Show more lines
📌 Keep only the top few chunks
(Too many would confuse the AI)

Prepare Containers 📦
Pythonparts: List[str] = []sources: List[Dict[str, Any]] = []``Show more lines
These are empty boxes:

parts → text sent to AI
sources → info shown to user later


Loop Through Each Chunk 🔄
Pythonfor index, chunk in enumerate(ctx_chunks, 1):Show more lines
📌 Go through chunks one by one
📌 Count them starting from 1

Confidence and Score
Pythonconf = chunk.get("confidence", "?")score = round(float(chunk.get("rerank_score", 0.0)), 4)Show more lines
Example:
confidence = high
score = 0.8732


Add Chunk to Context
Pythonparts.append("[Chunk 1 | confidence=high | score=0.8732]\nClause 3 says...")Show more lines
🧩 This helps the AI know:

where text came from
how trustworthy it is


Make a Short Preview 👀
Pythonpreview = chunk["text"][:150]``Show more lines
Only first 150 characters
Example:
"Clause 3 describes payment timelines..."

Add ... if longer ✂️

Save Source Info 📚
Pythonsources.append({    "index": index,    "text_preview": preview,    "confidence": conf,    "rerank_score": score,    "source_type": ...})``Show more lines
This helps later to show:

“Answer came from Chunk 2 (table, high confidence)”


Join All Chunks Together 🧵
Pythoncontext = "\n\n---\n\n".join(parts)Show more lines
📄 Makes one big document with separators

ADD CONVERSATION MEMORY 🧠 (Important Fix)
Pythonhistory_text = ""if history:Show more lines
If we had earlier chats…
Pythonhistory_text = "Previous conversation:\n"Show more lines
Loop over last 3 turns only:
PythonUser: What is clause 2?Assistant: Clause 2 is about paymentShow more lines
✅ This lets follow-ups like:

“What about clause 3?”


FINAL PROMPT SENT TO AI 🧠🤖
Pythonprompt = f"""Previous conversation: ...Context: ...Question: {query}Answer:"""Show more lines
This is what the AI actually sees

Call the AI Models 🔁
Pythonfor model in [PRIMARY_LLM, FALLBACK_LLM]:Show more lines
Try:

Primary brain 🧠
Backup brain 🧠🧠


Generate Answer
Pythoncandidate = llm_generate(...)Show more lines
If answer is good ✅:

Save it
Stop looping

If empty ❌:

Try fallback

If crash 💥:

Log error
Try next model


If Both Models Fail ❌❌
Pythonanswer = "Both LLMs failed to generate a response."flags.append("empty_answer")Show more lines

Hallucination Check 👻
Pythonfor phrase in HALLUCINATION_PHRASES:Show more lines
Looks for signs like:

“I think”
“Probably”
“Not sure but”

⚠️ Adds warning flag

Measure Time ⏱️
Pythonelapsed = round((time.time() - t0) * 1000, 1)Show more lines
Example:
245.3 ms


Final Return ✅🎁
Pythonreturn {    "answer": answer,    "model_used": model_used,    "fallback_used": fallback_used,    "guardrail_flags": flags,    "sources": sources,    "node_latencies": ...}``Show more lines
🎉 This is the final package sent back to user

PART 2: Helper Function 🧰
Pythondef _detect_chunk_type(text: str) -> str:``Show more lines
🕵️ Detective that guesses what kind of content this chunk is

Table Detection 📊
Pythonif "|" in text and text.count("|") > 3:``Show more lines
Lots of | → probably a table

Image Caption 🖼️
Pythonif text.strip().startswith("[Caption"):Show more lines
Starts like:
[Caption] Image shows...


Chart 📈
Pythonif "chart shows" in text.lower():``Show more lines
Words like:

figure
chart
graph
plot


Default
Pythonreturn "text"Show more lines
Normal paragraph ✍️

FINAL SIMPLE SUMMARY 🧒
This code:
✅ Protects against bad inputs
✅ Uses best documents only
✅ Remembers recent chat
✅ Tries two AI brains
✅ Tells you where the answer came from
✅ Measures speed
✅ Avoids hallucinations
If you want, I can:

Draw a flow diagram
Explain like LEGO blocks
Convert this into comments inside the code
Or explain how data flows step-by-step

Just tell me 🙂
Provide your feedback on BizChatYou said: _compiled_graph = None

def route_after_guardrail(state: AgentState):
    flags = state.get("guardrail_flags", [])
    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return "generator"
    return "expander"

def get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", node_guardrail)
    graph.add_node("expander", node_query_expander)
    graph.add_node("retriever", node_retriever)
    graph.add_node("reranker", node_reranker)
    graph.add_node("generator", node_generator)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"expander": "expander", "generator": "generator"},
    )
    graph.add_edge("expander", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", END)

    _compiled_graph = graph.compile()
    return _compiled_graph

def _initial_state(query: str, history: Optional[List] = None) -> AgentState:
    return {
        "original_query": query,
        "current_query": query,
        "query_variants": [query],
        "detected_intent": "unknown",
        "conversation_history": history or [],
        "raw_chunks": [],
        "ranked_chunks": [],
        "overall_confidence": "low",
        "answer": "",
        "model_used": "",
        "fallback_used": False,
        "guardrail_flags": [],
        "retry_count": 0,
        "node_latencies": {},
        "sources": [],
        "quality_gate_failed": False,
    }

def _prepare_retry_state(state: AgentState) -> Optional[AgentState]:
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return None

    flags = state.get("guardrail_flags", [])
    if "possible_hallucination" not in flags:
        return None

    pstatus(f"{C.YELLOW}Hallucination detected -> retry {retry_count + 1}{C.RESET}", C.YELLOW)
    new_query = (
        f"{state['original_query']} -- answer using ONLY exact facts stated in the document. Do not infer."
    )

    next_state: AgentState = {
        **state,
        "current_query": new_query,
        "query_variants": [new_query],
        "raw_chunks": [],
        "ranked_chunks": [],
        "sources": [],
        "answer": "",
        "model_used": "",
        "fallback_used": False,
        "quality_gate_failed": False,
        "retry_count": retry_count + 1,
        "guardrail_flags": [f for f in flags if f not in {"possible_hallucination", "empty_answer"}],
        "node_latencies": {},
    }
    return next_state

def run_agent(query: str, history: Optional[List] = None):
    graph = get_graph()
    state: AgentState = _initial_state(query, history=history)
    t0 = time.time()

    while True:
        result = graph.invoke(state)
        retry_state = _prepare_retry_state(result)
        if retry_state is None:
            result["wall_ms"] = round((time.time() - t0) * 1000)
            return result
        state = retry_state
now explain thisCopilot said: CopilotAlright 😊
I’ll explain this very slowly, simply, and playfully, like explaining a factory + traffic system to a kid.
No hard words without meaning, lots of examples.

Big Picture First 🗺️
This code builds and runs a smart question‑answer robot using:

🚦 Rules first (guardrail)
🔍 Search documents
🧠 Think with AI
🔁 Retry if it lies
🧰 Reuse the same graph again and again

Think of it like a theme park ride with stations and decisions.

1️⃣ _compiled_graph = None
Python_compiled_graph = None``Show more lines
Like a kid version 🧒
👉 “We don’t have the map yet.”

This is a global memory box
It will store the ready-made graph
So we don’t build the same thing again and again (fast 🚀)


2️⃣ route_after_guardrail
Pythondef route_after_guardrail(state: AgentState):Show more lines
What is this?
🚦 Traffic controller
It decides where to go next after safety checks.

Pythonflags = state.get("guardrail_flags", [])Show more lines
🧠 Look at warning stickers put earlier.
Examples:

"prompt_injection_detected"
"empty_query"


Pythonif "prompt_injection_detected" in flags or "empty_query" in flags:Show more lines
🛑 If someone:

tries to hack
or asks nothing at all


Pythonreturn "generator"Show more lines
👉 Skip searching → go straight to answer generator
(because we already know it’s bad)

Pythonreturn "expander"Show more lines
✅ Otherwise:

Expand the question
Search documents
Do the normal smart flow


🚦 In kid terms:

“If the question is naughty or empty → stop early
If it’s okay → keep going”


3️⃣ get_graph()
Pythondef get_graph():``Show more lines
🛠️ This builds the whole machine

Pythonglobal _compiled_graphShow more lines
📦 Use the global storage box

Pythonif _compiled_graph is not None:    return _compiled_graph``Show more lines
♻️ If already built:

“Reuse the toy you already made”


Build the Graph 🧩
Pythongraph = StateGraph(AgentState)Show more lines
🎢 Create a railway track

AgentState is the backpack that travels along


Add Stations 🚉
Pythongraph.add_node("guardrail", node_guardrail)Show more lines
🚦 Safety station
Pythongraph.add_node("expander", node_query_expander)Show more lines
🔍 Rewrite question in better ways
Pythongraph.add_node("retriever", node_retriever)Show more lines
📚 Find documents
Pythongraph.add_node("reranker", node_reranker)Show more lines
🥇 Pick best documents
Pythongraph.add_node("generator", node_generator)Show more lines
🤖 AI writes the answer

Connect Stations with Roads 🛣️
Pythongraph.add_edge(START, "guardrail")Show more lines
🟢 Everyone starts at safety 🚦

Conditional Road (Decision Point) 🔀
Pythongraph.add_conditional_edges(    "guardrail",    route_after_guardrail,    {"expander": "expander", "generator": "generator"},)Show more lines
🧠 After guardrail:

If unsafe → go to generator
If safe → go to expander

Just like:

“If raining → go inside
Else → go to playground”


Normal Flow 🧵
Pythongraph.add_edge("expander", "retriever")graph.add_edge("retriever", "reranker")graph.add_edge("reranker", "generator")graph.add_edge("generator", END)``Show more lines
➡️ One step after another
➡️ 끝 (END)

Python_compiled_graph = graph.compile()Show more lines
🔧 Lock everything in place
📦 Store it for later reuse

Pythonreturn _compiled_graph``Show more lines
✅ Done!

4️⃣ _initial_state
Pythondef _initial_state(query: str, history: Optional[List] = None)Show more lines
🍼 This creates a fresh baby state

What goes inside?
Python"original_query": queryShow more lines
User question:

“Explain clause 3”


Python"current_query": queryShow more lines
Current working version (may change later)

Python"query_variants": [query]``Show more lines
List of versions of the query

Python"conversation_history": history or []Show more lines
Previous chat memory 🧠

Python"raw_chunks": []"ranked_chunks": []Show more lines
📚 Empty document boxes

Python"guardrail_flags": []``Show more lines
🚨 No warnings yet

Python"retry_count": 0Show more lines
🔁 No retries yet

Python"quality_gate_failed": False``Show more lines
✅ Documents are okay (for now)

Summary 🧒

“This is a clean notebook before class starts”


5️⃣ _prepare_retry_state
Pythondef _prepare_retry_state(state: AgentState)``Show more lines
🔁 This decides:

“Should we try again because the AI made up stuff?”


Pythonretry_count = state.get("retry_count", 0)Show more lines
How many times did we already retry?

Pythonif retry_count >= MAX_RETRIES:    return NoneShow more lines
🛑 Too many retries → stop forever

Pythonflags = state.get("guardrail_flags", [])Show more lines
Check warnings again

Pythonif "possible_hallucination" not in flags:    return NoneShow more lines
✅ If AI didn’t lie → no retry needed

If AI lied 👻
Pythonpstatus("Hallucination detected -> retry", ...)Show more lines
📣 Log message

Pythonnew_query = (  "... answer using ONLY exact facts stated in the document")Show more lines
💢 Yell at AI:

“NO guessing. ONLY copy facts.”


Reset Stuff 🧹
Python"raw_chunks": [],"ranked_chunks": [],"sources": [],"answer": "",Show more lines
Clean everything

Python"retry_count": retry_count + 1Show more lines
Add retry number ➕

Pythonreturn next_stateShow more lines
🎁 New state → try again

6️⃣ run_agent
Pythondef run_agent(query: str, history: Optional[List] = None):Show more lines
🚀 This is the START BUTTON

Pythongraph = get_graph()Show more lines
Get the roller-coaster 🎢

Pythonstate = _initial_state(query, history)Show more lines
Fresh backpack 🎒

Pythont0 = time.time()Show more lines
⏱️ Start stopwatch

Infinite Loop 🔄
Pythonwhile True:Show more lines
Try → check → retry if needed

Pythonresult = graph.invoke(state)Show more lines
🎢 Send state through the whole graph

Pythonretry_state = _prepare_retry_state(result)Show more lines
Ask:

“Did the AI lie?”


Pythonif retry_state is None:Show more lines
✅ No retry needed

Pythonresult["wall_ms"] = ...return resultShow more lines
🏁 Stop and return final answer 🎉

Pythonstate = retry_stateShow more lines
🔁 Otherwise: retry with stricter rules

🧁 SUPER SIMPLE SUMMARY (Kid Level)
This code:
✅ Builds a question-answering factory
✅ Checks safety first 🚦
✅ Searches documents 📚
✅ Uses AI 🤖
✅ Detects lies 👻
✅ Retries if needed 🔁
✅ Reuses the same factory to be fast ⚡

If you want next, I can:

Draw a flow diagram
Explain as LEGO blocks
Explain how data looks at each step
Or guide you like “how to explain this in an interview”

Just say what you want 😊
Provide your feedback on BizChatYou said: def interactive_loop():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}ingest <path>{C.RESET}        Ingest a file (any format)
  {C.CYAN}ingest <p1> <p2> ...{C.RESET} Ingest multiple files
  {C.CYAN}stats{C.RESET}                Show chunk count in Milvus
  {C.CYAN}reset{C.RESET}                Clear the collection
  {C.CYAN}history{C.RESET}              Show current conversation memory
  {C.CYAN}clear{C.RESET}                Clear conversation memory
  {C.CYAN}quit{C.RESET}                 Exit
""")

    # FIX: conversation history persisted across turns in the session
    conversation_history: List[Dict] = []

    while True:
        try:
            user_input = input(f"\n{C.GREEN}{C.BOLD}Q: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}Session ended.{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{C.GRAY}Session ended.{C.RESET}")
            break

        if user_input.lower() == "reset":
            try:
                reset_collection()
            except Exception as exc:
                perr(f"Reset failed: {exc}")
            continue

        if user_input.lower() == "clear":
            conversation_history = []
            pok("Conversation memory cleared.")
            continue

        if user_input.lower() == "history":
            if not conversation_history:
                pstatus("No conversation history yet.", C.GRAY)
            for i, turn in enumerate(conversation_history, 1):
                print(f"  {C.BLUE}Turn {i}{C.RESET}")
                print(f"    Q: {turn.get('query', '')[:80]}")
                print(f"    A: {turn.get('answer', '')[:80]}...")
            continue

        if user_input.lower().startswith("ingest "):
            paths = user_input[7:].strip().split()
            valid = [path for path in paths if os.path.isfile(path)]
            for path in paths:
                if not os.path.isfile(path):
                    perr(f"File not found: {path}")
            if valid:
                try:
                    run_ingest(valid)
                except Exception as exc:
                    perr(f"Ingest failed: {exc}")
                    logger.exception("Ingest failed")
            continue

        if user_input.lower() == "stats":
            try:
                milvus = get_milvus()
                stats = milvus.get_collection_stats(COLLECTION)
                pok(f"Collection: {COLLECTION}")
                pstatus(f"Stats: {json.dumps(stats, indent=2)}", C.GRAY)
            except Exception as exc:
                perr(f"Stats failed: {exc}")
            continue

        pstatus(f"Query: {C.WHITE}{user_input}{C.RESET}")
        if conversation_history:
            pstatus(f"Memory: {len(conversation_history)} turn(s) in context", C.GRAY)
        print()

        try:
            result = run_agent(user_input, history=conversation_history)

            # Update conversation memory
            if result.get("answer") and result.get("model_used") != "none":
                conversation_history.append({
                    "query": user_input,
                    "answer": result["answer"],
                })
                # Keep last 10 turns to avoid unbounded growth
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

            print_answer(
                result.get("answer", ""),
                result.get("overall_confidence", "low"),
                result.get("wall_ms", 0),
                result.get("model_used", "?"),
                result.get("retry_count", 0),
                result.get("sources", []),
                result.get("node_latencies", {}),
                result.get("guardrail_flags", []),
            )
        except Exception as exc:
            perr(f"Agent failed: {exc}")
            logger.exception("Agent failed")
explain this tooCopilot said: CopilotAlright 😊
I’ll explain this just like before — slow, simple, friendly, and like explaining a game to a kid.
I’ll go section by section and tell you what it does, why it’s there, and give tiny examples.

🧠 Big Picture First (Very Simple)
This function is an interactive chat loop.
Think of it like a talking robot sitting in a terminal where you can:

Ask questions ❓
Upload files 📄
See memory 🧠
Clear memory 🧹
Reset database 🔄
Quit 👋

It keeps running until you say “quit”.

1️⃣ Function Start
Pythondef interactive_loop():``Show more lines
👶 Meaning:

“Start the never‑ending talking session.”

This is the main chat loop for your assistant.

2️⃣ Show Commands Menu 📜
Pythonprint(f"""Commands:  ingest <path>  stats  reset  history  clear  quit""")Show more lines
🧒 Like showing a menu in a video game:
🎮 You can type:

ingest file.pdf → give documents
stats → see database info
reset → wipe database
history → show chat memory
clear → forget conversation
quit → exit game

The colors (C.BOLD, C.CYAN) are just pretty text 🌈

3️⃣ Conversation Memory 🧠
Pythonconversation_history: List[Dict] = []Show more lines
📦 This is a memory box.
Example inside:
Python[  {"query": "What is clause 3?", "answer": "Clause 3 is about payments"}]``Show more lines
✅ This memory:

Helps follow‑up questions
Is kept during this session only


4️⃣ Infinite Listening Loop 🔁
Pythonwhile True:``Show more lines
🧒 Meaning:

“Keep listening forever (until user quits)”


5️⃣ Take User Input 🎙️
Pythonuser_input = input("Q: ").strip()Show more lines
👂 Wait for user to type something.
Example:
Q: what does clause 3 mean


Exit Safely 🚪
Pythonexcept (EOFError, KeyboardInterrupt):``Show more lines
If user presses:

Ctrl+D
Ctrl+C

👉 Exit nicely instead of crashing.

6️⃣ Empty Input ❌
Pythonif not user_input:    continue``Show more lines
If user presses Enter without typing
→ Ignore and ask again.

7️⃣ Quit Commands 👋
Pythonif user_input.lower() in ("quit", "exit", "q"):Show more lines
If user types:
quit
exit
q

👉 Print goodbye and stop loop.

8️⃣ Reset Database 🔄
Pythonif user_input.lower() == "reset":Show more lines
🧹 Clears all stored document chunks.
Pythonreset_collection()``Show more lines
Think:

“Delete all books from the library.”


9️⃣ Clear Conversation Memory 🧠🧹
Pythonif user_input.lower() == "clear":``Show more lines
✅ Clears chat history only
📚 Documents stay intact
Pythonconversation_history = []Show more lines
Example:

Assistant forgets earlier answers.


🔟 Show History 📜
Pythonif user_input.lower() == "history":``Show more lines
Prints conversation memory.
Example output:
Turn 1
  Q: What is clause 3?
  A: Clause 3 talks about payments...

If empty:

“No conversation history yet.”


1️⃣1️⃣ Ingest Files 📄➕
Pythonif user_input.lower().startswith("ingest "):``Show more lines
When user types:
ingest contract.pdf invoice.docx


Split Paths
Pythonpaths = user_input[7:].strip().split()Show more lines
Creates list:
Python["contract.pdf", "invoice.docx"]Show more lines

Check If Files Exist ✅❌
Pythonvalid = [path for path in paths if os.path.isfile(path)]``Show more lines

✅ real files → ingest
❌ fake files → error message


Ingest Files
Pythonrun_ingest(valid)``Show more lines
📥 This:

reads files
chunks them
stores them in vector DB


1️⃣2️⃣ Show Database Stats 📊
Pythonif user_input.lower() == "stats":``Show more lines
Gets info from Milvus (vector database).
Example output:
Python{  "row_count": 321,  "segments": 5}Show more lines
👶 Like saying:

“Library has 321 pages stored.”


1️⃣3️⃣ REAL QUESTION TIME ❓🤖
If none of the commands matched, then:
Pythonpstatus(f"Query: {user_input}")Show more lines
✅ This is a real question for the AI.

Show Memory Info
Pythonif conversation_history:    pstatus("Memory: 2 turns in context")Show more lines
🧠 Shows how much chat memory will be used.

1️⃣4️⃣ Call the Agent 🚀
Pythonresult = run_agent(user_input, history=conversation_history)Show more lines
🎢 Sends question through entire graph:

Guardrail 🚦
Expander 🔍
Retriever 📚
Reranker 🥇
Generator 🤖

Receives:

answer
sources
timings
flags


1️⃣5️⃣ Save Conversation Memory 🧠
Pythonif result.get("answer") and result.get("model_used") != "none":Show more lines
✅ Only save real answers, not errors.
Pythonconversation_history.append({  "query": user_input,  "answer": result["answer"]})Show more lines

Limit Memory Size 🧠✂️
Pythonif len(conversation_history) > 10:    conversation_history = conversation_history[-10:]``Show more lines
✅ Keeps last 10 turns only
🚫 Avoids memory explosion

1️⃣6️⃣ Print Final Answer 🖨️✨
Pythonprint_answer(...)Show more lines
Shows:

✅ Answer text
🧠 Confidence
⏱️ Time taken
🤖 Model used
🔁 Retry count
📚 Sources
⚠️ Guardrail flags

This is what the user sees.

1️⃣7️⃣ Catch Errors 💥
Pythonexcept Exception as exc:Show more lines
If anything breaks:

Print error
Log details
Don’t crash app ✅


🧁 SUPER SIMPLE SUMMARY (Kid Version)
This function is like:
🗣️ A talking robot console that:

waits for you
understands commands
remembers chats
answers questions
reads your documents
stays safe
never crashes easily

You can:
ingest files → ask questions → see answers → follow up → quit


If you want next, I can:

Explain how all 3 files connect together
Draw the full system flow
Explain like interview explanation
Or help you refactor this into cleaner code

Just tell me 🙂
