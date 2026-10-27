# 🎯 LangGraph Agent & RAG: The Ultimate Interview Playbook

This guide contains everything you need to confidently discuss, explain, and defend this project in AI/LLM Engineering interviews.

---

## 1. The 30-Second Elevator Pitch

> *"I designed and built a multi-session conversational AI agent using **LangGraph**, **ChromaDB**, **SQLite**, and **Streamlit**. The system implements a cyclic **ReAct state machine** capable of dynamically deciding when to answer directly, perform web searches via DuckDuckGo, compute arithmetic, or execute thread-isolated RAG over user-uploaded PDF documents.*
>
> *Key architectural highlights include **thread-isolated document scoping** so documents from one chat never leak into another, **LangGraph SQLite state checkpointing** for persistent multi-turn memory, and **native digital text stream extraction** rather than heavy OCR to ensure fast, deterministic document processing. I also built live token streaming with dynamic tool-execution status in Streamlit."*

---

## 2. End-to-End System Data Flow

```
[User Input in Streamlit]
          │
          ▼
[Runtime Config: thread_id injected]
          │
          ▼
[LangGraph StateGraph: schema(messages)]
          │
          ▼
┌──────────────────┐
│   chatbot Node   │◄─────────────────────────────┐
│ (LLM with tools) │                              │
└─────────┬────────┘                              │
          │                                       │
          ▼                                       │
   [tools_condition]                              │
    /             \                               │
[Tool Calls?]      [No Tool Calls]                │
    /                     \                       │
  YES                      NO                     │
  /                         \                     │
 ▼                           ▼                    │
┌──────────────┐         [END Node]               │
│  ToolNode    │       (Stream to UI)             │
└──────┬───────┘                                  │
       │                                          │
       ├───> [rag_tool] ──> Filter ChromaDB by thread_id ──> Return Context & Citations ──┘
       ├───> [web_search] ──> DuckDuckGo API ──> Return Web Results ──────────────────────┘
       └───> [calculator_tool] ──> Python Arithmetic ──> Return Result ───────────────────┘
```

---

## 3. Core Architectural Decisions (What to Tell Interviewers)

### A. Why LangGraph over Vanilla LangChain or Simple Loops?
1. **Cyclic ReAct Control**: Vanilla LangChain chains (LCEL) are Directed Acyclic Graphs (DAGs); they flow in one direction. Real-world agents require **cycles**—the LLM calls a tool, inspects the output, potentially corrects itself or calls another tool, and then synthesizes an answer (`chatbot -> tools -> chatbot`).
2. **First-Class State & Checkpointing**: LangGraph persists intermediate graph state to SQLite (`SqliteSaver`) keyed by `thread_id`. If the server restarts or the user returns days later, the entire message history and agent execution trace resume seamlessly.
3. **Controllability**: Unlike black-box frameworks (e.g., CrewAI or AutoGen), LangGraph gives you full explicit control over state schemas, conditional edges, and human-in-the-loop inspection points.

### B. Why Native PDF Extraction (`PyPDFLoader`) instead of OCR?
*When asked: "Why didn't you use OCR?"*
1. **Target Document Type**: Digital PDFs (research papers, reports, study material) already contain structured text streams.
2. **Speed & Latency**: Native digital extraction with `pypdf` extracts pages in **milliseconds**, whereas OCR (like Tesseract or cloud vision APIs) requires rendering pages to images and takes **seconds per page**.
3. **Deterministic Accuracy**: Native extraction extracts the exact encoded characters with 100% fidelity. OCR introduces optical noise, character misspellings, and high compute overhead.
4. **Clean Deployment**: OCR requires heavy external C-libraries (Leptonica, Tesseract) or expensive multimodal vision models. Native extraction has zero heavy dependencies.
> *(Tip: If the interviewer asks "What if a user uploads a scanned image PDF?", answer: "We would use a preprocessing pipeline where we check if extracted character count is near zero. Only if it's a scanned image would we route it through a vision/OCR fallback.")*

### C. Thread-Isolated Multi-Tenant RAG (Preventing Data Leaks)
*When asked: "How do you ensure Document A in Chat 1 isn't seen by Chat 2?"*
1. When a PDF is ingested, we generate a unique `document_id` and store the link `(thread_id, document_id)` in SQLite.
2. Every chunk stored in ChromaDB has `document_id` in its metadata.
3. When `rag_tool` executes, it looks up the active `thread_id` and queries SQLite for document IDs belonging *only* to that thread.
4. ChromaDB performs a filtered similarity search:
   ```python
   vector_store.similarity_search(query, k=4, filter={"document_id": {"$in": thread_doc_ids}})
   ```
   This guarantees mathematical isolation between different user conversations without needing separate vector databases.

### D. Injected Runtime Configuration (`RunnableConfig`)
Instead of making the LLM remember or guess the UUID `thread_id`, we pass `RunnableConfig` to `rag_tool`. LangGraph automatically injects the active thread context from runtime:
```python
@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id")
```
This eliminates hallucination and guarantees secure, reliable thread resolution.

### E. Corrective RAG (CRAG) with Autonomous Web Fallback
*When asked: "What happens when retrieved document passages don't have the answer?"*
1. **The Flaw of Naive RAG**: In naive RAG, if ChromaDB returns zero or irrelevant chunks, the model either hallucinates or gives a dead-end response.
2. **The Corrective Solution**: In our agent, `rag_tool` evaluates document match quality. If context is insufficient, it signals `insufficient_context`.
3. **Self-Healing Loop**: The agent's prompt policy dictates that upon detecting insufficient document context, it immediately invokes `web_search` (DuckDuckGo) as a corrective action.
4. **Transparency**: The final answer cites the web sources and clarifies that the document lacked this specific information, ensuring complete user transparency.

### F. RAG Pipeline Evaluation (The RAG Triad Benchmark)
*When asked: "How did you systematically evaluate your RAG pipeline?"*
We implemented an evaluation engine based on the **RAG Triad** (the core methodology behind Ragas & TruLens) using LLM-as-a-judge:
1. **Faithfulness (Groundedness) [1.0 - 5.0]**: Measures whether every factual claim in the generated answer is supported by the retrieved context chunks (penalizes hallucinations).
2. **Answer Relevance [1.0 - 5.0]**: Measures how completely and directly the generated answer addresses the user's prompt (penalizes evasive or tangential answers).
3. **Context Precision [1.0 - 5.0]**: Measures the signal-to-noise ratio of the retrieved vector chunks (penalizes retrieval of irrelevant passages).
*Evaluation Delivery*:
- **Interactive UI**: Users can click *"📊 Evaluate Answer Quality"* in Streamlit to see real-time scorecards and reasoning.
- **Offline Benchmark (`evaluate_rag.py`)**: An automated test runner that benchmarks test scenarios and produces an overall performance scorecard.

---

## 4. Top 15 Technical Interview Questions & Answers

### Q1: How does LangGraph maintain state across turns?
**Answer:** LangGraph defines an `AgentState` schema with an `Annotated[list[BaseMessage], add_messages]` field. State is persisted using a checkpointer (`SqliteSaver`). Each turn is passed a `config={"configurable": {"thread_id": "<id>"}}`. The checkpointer loads the latest state snapshot for that thread ID, executes the graph nodes, and saves the new snapshot to SQLite upon completion.

### Q2: What is the `add_messages` reducer function?
**Answer:** In LangGraph, when a node returns a dictionary like `{"messages": [new_message]}`, the reducer specifies *how* to combine the new value with the existing state. Default assignment would overwrite the entire list. `add_messages` acts as an append-and-update reducer: it appends new messages while updating existing messages if their `id` matches (useful for streaming or tool message replacement).

### Q3: What is the difference between Router Agent and ReAct Agent?
**Answer:** A router agent classifies user intent once and directs the query to a single static branch (e.g., either RAG or Web Search). A ReAct (Reason + Act) agent operates in a loop: the LLM reason about what tool to call, inspects the result, and can choose to invoke additional tools or finish, handling multi-step queries (e.g., "Look up the GDP of France from the PDF, then calculate 15% of it").

### Q4: Why did you pick chunk size 1000 with 200 overlap?
**Answer:** A chunk size of 1000 characters (~200–250 tokens) is optimal for general textual documents because it captures complete coherent paragraphs without diluting semantic meaning. A 200-character overlap (~40–50 tokens) prevents losing crucial context or split sentences across chunk boundaries.

### Q5: What embedding model did you use and what is its dimensionality?
**Answer:** By default, the system uses OpenAI's `text-embedding-3-small`, which outputs **1536-dimensional** dense vectors normalized for cosine similarity. It also supports local embeddings via Ollama (`nomic-embed-text`, 768 dimensions) for completely offline/private execution.

### Q6: What vector distance metric is used in ChromaDB?
**Answer:** ChromaDB defaults to **Cosine Distance** (or Squared L2). With normalized embeddings like `text-embedding-3-small`, cosine similarity is directly proportional to the dot product of the vectors, providing fast semantic nearest-neighbor search.

### Q7: How does the agent prevent infinite tool execution loops?
**Answer:** Two safeguards:
1. **LangGraph Recursion Limit**: LangGraph has a built-in `recursion_limit` parameter (defaults to 25 steps). If a graph exceeds this count, it raises a `GraphRecursionError`.
2. **Tool Output Clarity**: The tools return clear status messages (e.g., if no document is found or no relevant passages match), giving the LLM sufficient context to synthesize a direct response rather than retrying.

### Q8: How does token streaming work alongside tool calls in Streamlit?
**Answer:** We call `workflow.stream(..., stream_mode="messages")`. In this mode, LangGraph yields message chunks as they are generated. If a `ToolMessage` chunk is encountered, we update a Streamlit `st.status` widget showing that an external action is executing. When `AIMessage` content chunks arrive, they are yielded directly to `st.write_stream` for smooth, responsive user-facing streaming.

### Q9: How does the automatic title generator work without slowing down the chat?
**Answer:** After the first user-assistant interaction in a "New Chat", the frontend triggers a lightweight naming chain (`PromptTemplate | model | PydanticOutputParser`). It inspects the first 4 messages, summarizes them into a 3-5 word title, updates SQLite, and refreshes the sidebar seamlessly.

### Q10: What happens if ChromaDB returns zero relevant chunks for a query?
**Answer:** `rag_tool` checks the result list. If empty, it returns a structured JSON payload: `{"status": "no_match", "message": "Searched the uploaded document(s), but found no closely relevant passages."}`. The LLM receives this in the next turn and informs the user honestly, avoiding hallucinations.

### Q11: How do you handle database connection locking in SQLite with multiple requests?
**Answer:** We connect with `sqlite3.connect(DATABASE_NAME, check_same_thread=False)` and wrap database transactions in Python context managers (`with get_connection() as conn:`). SQLite runs in WAL (Write-Ahead Logging) mode, enabling concurrent readers while a write occurs.

### Q12: How would you scale this architecture for 100,000 concurrent users?
**Answer:**
1. **State Persistence**: Replace `SqliteSaver` with `PostgresSaver` (PostgreSQL with connection pooling via PgBouncer).
2. **Vector Store**: Transition from local embedded ChromaDB to a managed distributed vector database like **Pinecone**, **Milvus**, or **pgvector**.
3. **Worker Decoupling**: Run LangGraph instances in stateless containerized workers (FastAPI / Celery / Temporal) behind an API Gateway, with Redis for pub/sub message streaming.
4. **Caching**: Implement semantic caching (e.g., GPTCache or Redis Vector Cache) for identical or highly similar user queries.

### Q13: How would you evaluate the quality of this RAG pipeline?
**Answer:** Using the **Ragas** or **TruLens** evaluation framework against an evaluation dataset:
- **Faithfulness**: Are answers grounded purely in retrieved context?
- **Answer Relevance**: Did the agent actually address the user's prompt?
- **Context Precision & Recall**: Did Chroma retrieve the true ground-truth chunks?

### Q14: How does the system defend against prompt injection via uploaded PDFs?
**Answer:** Document text retrieved by RAG is returned to the model inside a structured tool output (`ToolMessage`), not as system instructions. The LLM's system prompt explicitly instructs it to treat retrieved document content as reference context rather than executable instructions. In production, we can also add an input/output guardrail layer (such as NeMo Guardrails or Llama Guard).

### Q15: Why separate `threads`, `documents`, and `thread_documents` tables in SQLite?
**Answer:** This implements a normalized many-to-many schema. A conversation thread can have multiple uploaded documents, and if desired in the future, a single indexed document could be linked across multiple threads without re-chunking and duplicating vector embeddings in ChromaDB.

### Q16: Why did you implement Model Context Protocol (MCP) in this project?
**Answer:** MCP (developed by Anthropic) is rapidly becoming the universal open standard for connecting LLMs to data sources and tools, replacing fragmented, proprietary plugin APIs. By implementing MCP:
1. **Universal Interoperability**: External client tools (Claude Desktop, Cursor, Antigravity, or other agent orchestrators) can immediately connect to our LangGraph agent and use our RAG pipeline, PDF ingestion, and arithmetic tools via standard JSON-RPC.
2. **Standardized Context Protocol**: MCP defines strict schemas for Tools, Resources (URIs like `rag://threads`), and Prompts, ensuring clean contract validation without custom REST boilerplate.
3. **Dual Role Architecture**: Our project acts as an **MCP Server** (sharing our capabilities with the world) and an **MCP Client** (dynamically mounting external tools into our LangGraph ReAct agent).

### Q17: What are the differences between MCP stdio and SSE transports?
**Answer:**
- **`stdio` (Standard Input/Output)**: The client launches the MCP server as a child process and communicates via stdin/stdout streams. This is the zero-overhead, ultra-secure standard for local desktop clients (Claude Desktop, Cursor) because it runs locally with no open network ports or authentication burden.
- **`sse` (Server-Sent Events over HTTP)**: The server runs as a standalone HTTP service and streams events to remote clients. This is ideal for distributed architectures, microservices, and containerized deployments (e.g. Kubernetes, Docker) where multiple clients connect to a shared agent service over the network.

### Q18: How does your MCP server preserve thread isolation and multi-tenancy?
**Answer:** Tools exposed over MCP (such as `search_rag_context` and `chat_agent_turn`) accept a `thread_id` parameter. Vector searches perform metadata filtering against SQLite thread-document relationships, guaranteeing that even when invoked through an external MCP client, queries never leak document passages across different conversation threads.

---

## 5. Quick Code Tour (Where to Look During a Screen Share)

| Feature | File & Function |
| :--- | :--- |
| **Model Context Protocol Server** | [`mcp_server.py: MCPServer`](file:///Users/harshsinghal/Desktop/llm/Chatbot/mcp_server.py) |
| **Dynamic MCP Client Adapter** | [`backend/mcp_client.py: get_dynamic_mcp_tools`](file:///Users/harshsinghal/Desktop/llm/Chatbot/backend/mcp_client.py) |
| **MCP Test Harness** | [`test_mcp.py: run_mcp_tests`](file:///Users/harshsinghal/Desktop/llm/Chatbot/test_mcp.py) |
| **LangGraph Definition** | [`backend/backend.py: StateGraph`](file:///Users/harshsinghal/Desktop/llm/Chatbot/backend/backend.py) |
| **Injected Config RAG Tool** | [`backend/backend.py: rag_tool`](file:///Users/harshsinghal/Desktop/llm/Chatbot/backend/backend.py) |
| **Digital PDF Ingestion** | [`backend/backend.py: ingest_pdf`](file:///Users/harshsinghal/Desktop/llm/Chatbot/backend/backend.py) |
| **SQLite Schema & Helpers** | [`backend/backend.py: initialize_database`](file:///Users/harshsinghal/Desktop/llm/Chatbot/backend/backend.py) |
| **Token Streaming UI & MCP Panel** | [`frontend/frontend.py: stream_agent_response`](file:///Users/harshsinghal/Desktop/llm/Chatbot/frontend/frontend.py) |
| **RAG Triad Evaluator** | [`evaluator.py: evaluate_rag_turn`](file:///Users/harshsinghal/Desktop/llm/Chatbot/evaluator.py) |
| **Automated RAG Benchmark** | [`evaluate_rag.py: TEST_CASES`](file:///Users/harshsinghal/Desktop/llm/Chatbot/evaluate_rag.py) |
